"""HTTP 边界：对外提供问答与元信息接口，并托管前端静态资源。

本模块只做三件事：把请求翻译成 `pipeline.ask` 的调用、把 `AnswerResult`
翻译成 JSON、把异常翻译成可读的中文提示。不含任何业务判断——检索、门控、
引用解析全部留在 pipeline.py 里。
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Config
from .embedder import EmbeddingError
from .index_store import IndexStore, IndexStoreError
from .llm import LLMError
from .pipeline import ask
from .prompts import section_label
from .reranker import RerankError

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
QUESTION_MAX = 500

# 图片白名单。不收 svg：svg 可内嵌脚本，即使作为 <img> 不执行也没有纳入的理由。
# 显式写 media type 而不是靠 mimetypes 猜——标准库不认识 .webp（返回 None）。
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
IMAGE_SUFFIXES = frozenset(IMAGE_TYPES)

# 图片路径的形状：以知识库为根的相对路径，且路径里必须出现 assets/ 这一层。
# 站点的目录层级不固定（<站点>/<栏目>/…/<章节>/assets/…），所以只校验「有 assets 这一层」；
# 真正的越界拦截由 _resolve_asset 里 resolve 之后再判界负责，形状检查只是第一道粗筛。
ASSET_SEGMENT = "/assets/"
ASSET_MISSING = "资源不存在"


class AskRequest(BaseModel):
    """一次提问。"""

    question: str = Field(min_length=1, max_length=QUESTION_MAX)
    sections: list[str] | None = None
    debug: bool = False


class CitationOut(BaseModel):
    """答案里的一条引用。excerpt 是父块原文全量，截断交由前端。"""

    index: int
    page_id: str  # 文章标识；界面据此定位图片所在目录
    title: str
    section: str
    section_label: str
    source_url: str
    heading_path: str
    excerpt: str


class AskResponse(BaseModel):
    """AnswerResult 的镜像，字段一一对应。"""

    answer: str
    rejected: bool
    citations: list[CitationOut]
    debug: dict


class SectionOut(BaseModel):
    """栏目的内部标识与展示名。key 是「站点/栏目」复合键，直接用作检索范围取值。"""

    key: str
    label: str


class SiteOut(BaseModel):
    """一个站点及其下属栏目。界面据此把筛选选项分组。"""

    key: str
    label: str
    sections: list[SectionOut]


class MetaResponse(BaseModel):
    """知识库元信息。只含模型名，不含任何密钥材料。"""

    page_count: int
    child_count: int
    parent_count: int
    built_at: str
    embedding_model: str
    rerank_model: str
    chat_model: str
    sites: list[SiteOut]


def _meta_payload(cfg: Config, store: IndexStore) -> MetaResponse:
    """索引元信息 + 查询期配置，组装成前端要的元信息。"""
    meta = store.meta
    site_labels = meta.get("site_labels") or {}
    section_labels = meta.get("section_labels") or {}

    grouped: dict[str, list[SectionOut]] = {}
    for key in store.sections:
        site = key.split("/", 1)[0]
        grouped.setdefault(site, []).append(
            SectionOut(key=key, label=section_label(key, section_labels))
        )

    # site_labels 的键序就是索引里站点的声明顺序（JSON 对象保序），排在前面的站点先展示；
    # 元信息没有站点名时退回字典序。
    order = [site for site in site_labels if site in grouped] or sorted(grouped)

    return MetaResponse(
        page_count=int(meta.get("page_count", 0)),
        child_count=len(store.entries),
        parent_count=int(meta.get("parent_count", 0)),
        built_at=str(meta.get("built_at", "未知")),
        embedding_model=str(meta.get("embedding_model", cfg.models.embedding_model)),
        rerank_model=cfg.models.rerank_model,
        chat_model=cfg.models.chat_model,
        sites=[
            SiteOut(
                key=site,
                label=site_labels.get(site, site),
                sections=grouped[site],
            )
            for site in order
        ],
    )


def _public_error(exc: Exception, cfg: Config) -> str:
    """异常文本里若混入密钥材料，一律替换后再交给客户端。"""
    text = str(exc)
    for secret in (cfg.models.embedding_api_key, cfg.models.chat_api_key):
        if secret:
            text = text.replace(secret, "***")
    return text


def _validation_message(exc: RequestValidationError) -> str:
    """把校验失败翻译成中文提示，替掉 Pydantic 的英文详情。"""
    kinds = {error.get("type") for error in exc.errors()}
    if "string_too_long" in kinds:
        return f"问题过长，请精简到 {QUESTION_MAX} 字以内"
    return "问题不能为空"


def _resolve_asset(cfg: Config, raw: str) -> tuple[Path, str]:
    """把请求路径解析成知识库内的图片文件，返回（路径, media type）。

    四道检查缺一不可：形状必须是知识库内的相对路径且含 assets/ 一层、扩展名在白名单内、
    resolve 之后仍落在知识库里（展开 .. 与符号链接再判界）、确实是文件。
    四类失败共用同一句话与同一状态码——区分「路径非法」和「文件不存在」
    等于告诉探测者哪个目录是存在的。
    """
    if raw.startswith("/") or ASSET_SEGMENT not in raw:
        raise HTTPException(status_code=404, detail=ASSET_MISSING)

    suffix = Path(raw).suffix.lower()
    if suffix not in IMAGE_TYPES:
        raise HTTPException(status_code=404, detail=ASSET_MISSING)

    # 必须先 resolve 再判界：字符串层面 a/../../etc/passwd 看着还在库里
    root = Path(cfg.kb_dir).resolve()
    target = (root / raw).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(status_code=404, detail=ASSET_MISSING)

    return target, IMAGE_TYPES[suffix]


def create_app(cfg: Config, store: IndexStore) -> FastAPI:
    """组装应用。索引已在调用方加载，这里只持有引用。"""
    # 关掉自动文档：Swagger UI 的静态资源来自 CDN，本机自用的工具不该依赖外网
    app = FastAPI(title="小林面试笔记 RAG 问答", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(RequestValidationError)
    def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": _validation_message(exc)})

    @app.get("/api/meta", response_model=MetaResponse)
    def read_meta() -> MetaResponse:
        return _meta_payload(cfg, store)

    @app.post("/api/ask", response_model=AskResponse)
    def read_answer(payload: AskRequest) -> AskResponse:
        try:
            result = ask(
                payload.question,
                cfg,
                store,
                sections=payload.sections or None,
                debug=payload.debug,
            )
        except (EmbeddingError, RerankError, LLMError, ValueError) as exc:
            # 先擦除再落日志与回传：日志同样会把文本带出本机
            message = _public_error(exc, cfg)
            logger.warning("问答失败：%s", message)
            raise HTTPException(status_code=502, detail=message) from exc

        return AskResponse(
            answer=result.answer,
            rejected=result.rejected,
            citations=[CitationOut(**asdict(citation)) for citation in result.citations],
            debug=result.debug,
        )

    @app.get("/kb/{path:path}")
    def read_asset(path: str) -> FileResponse:
        target, media_type = _resolve_asset(cfg, path)
        return FileResponse(target, media_type=media_type)

    # 静态挂载放最后：/api/* 与 /kb/* 先注册，路由优先匹配；html=True 让 / 直接返回 index.html
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


def serve(cfg: Config) -> None:
    """加载索引并启动服务。索引缺失时给出与 build 呼应的可读引导。"""
    if not IndexStore.exists(cfg.index_dir):
        raise IndexStoreError(
            f"索引尚未构建（{cfg.index_dir}）；请先运行 `xiaolinrag build` 再启动界面"
        )
    store = IndexStore.load(cfg.index_dir)
    uvicorn.run(
        create_app(cfg, store),
        host=cfg.webui.server_name,
        port=cfg.webui.server_port,
    )

"""配置加载与校验：config.yaml（非敏感）+ .env（密钥）。

密钥读取优先级：环境变量 > .env 文件，两者都缺则报可读错误（N5）。
任何对外输出走 redacted()，不泄漏完整密钥（N2）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

EMBEDDING_KEY_ENV = "SILICONFLOW_API_KEY"
CHAT_KEY_ENV = "CHAT_API_KEY"


class ConfigError(Exception):
    """配置缺失或非法，消息面向使用者可直接阅读。"""


def redact(secret: str) -> str:
    """密钥打码：保留前 6 位便于辨认，其余不输出。"""
    if not secret:
        return "(未设置)"
    return secret[:6] + "***"


@dataclass
class ModelConfig:
    embedding_base_url: str
    embedding_model: str
    rerank_base_url: str
    rerank_model: str
    chat_base_url: str
    chat_model: str
    judge_model: str
    chat_temperature: float
    embedding_api_key: str
    chat_api_key: str
    # 对话网关要求的额外请求头（如某些网关需要客户端声明会话标识）。为空即不加。
    chat_extra_headers: dict[str, str] = field(default_factory=dict)

    def redacted(self) -> dict:
        return {
            "embedding_model": self.embedding_model,
            "rerank_model": self.rerank_model,
            "chat_model": self.chat_model,
            "judge_model": self.judge_model,
            "embedding_api_key": redact(self.embedding_api_key),
            "chat_api_key": redact(self.chat_api_key),
        }


@dataclass
class ChunkingConfig:
    child_target: int
    child_max: int
    parent_target: int
    parent_max: int
    context_max_chars: int


@dataclass
class ContextualConfig:
    enabled: bool
    concurrency: int
    interval_seconds: float


@dataclass
class RetrievalConfig:
    vec_top_k: int
    bm25_top_k: int
    rrf_k: int
    rerank_top_n: int
    final_n: int
    gate_threshold: float


@dataclass
class WebUIConfig:
    server_name: str
    server_port: int


@dataclass
class MultiQueryConfig:
    """多 Query 扩展：检索前把 query 拆成多个角度分别检索（docs 12 方法四 / 13 第三路）。

    enabled 为假时走原单 query 链路，与现状逐字节一致；num_queries 是检索 query 总数
    （含原始 query，docs 14 要求原 query 必须保留，恒在首位）。默认值等于 config.yaml
    生产配置；load_config 总会显式填满，默认值只服务于直接构造 Config 的测试。
    """

    enabled: bool = True
    num_queries: int = 4
    temperature: float = 0.2


@dataclass
class Config:
    kb_dir: Path
    index_dir: Path
    models: ModelConfig
    chunking: ChunkingConfig
    contextual: ContextualConfig
    retrieval: RetrievalConfig
    webui: WebUIConfig
    multi_query: MultiQueryConfig = field(default_factory=MultiQueryConfig)

    def redacted(self) -> dict:
        """供 status 与日志使用的配置摘要，密钥已打码。"""
        return {
            "kb_dir": str(self.kb_dir),
            "index_dir": str(self.index_dir),
            "models": self.models.redacted(),
            "chunking": vars(self.chunking),
            "contextual": vars(self.contextual),
            "retrieval": vars(self.retrieval),
            "webui": vars(self.webui),
            "multi_query": vars(self.multi_query),
        }


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"配置缺少 `{name}` 小节，请对照 config.example.yaml 补齐")
    return value


def _require(section: dict, key: str, where: str):
    if key not in section or section[key] in (None, ""):
        raise ConfigError(f"配置 `{where}.{key}` 缺失或为空")
    return section[key]


def _resolve(path_value: str, base: Path, where: str) -> Path:
    if not path_value:
        raise ConfigError(f"配置 `{where}` 不能为空")
    p = Path(path_value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _extra_headers(models_section: dict) -> dict[str, str]:
    """读取对话网关的额外请求头，缺省为空（绝大多数网关不需要）。"""
    raw = models_section.get("chat_extra_headers") or {}
    if not isinstance(raw, dict):
        raise ConfigError("配置 `models.chat_extra_headers` 应为键值对")
    return {str(key): str(value) for key, value in raw.items()}


def load_config(path: str | Path | None = None) -> Config:
    """读取并校验配置。path 为 config.yaml 路径，密钥从同目录 .env 或环境变量取。"""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(
            f"未找到配置文件 {cfg_path}；请复制 config.example.yaml 为 config.yaml 后填写"
        )

    file_env = dotenv_values(cfg_path.parent / ".env")

    def secret(name: str) -> str:
        value = os.environ.get(name) or file_env.get(name) or ""
        if not value.strip():
            raise ConfigError(
                f"未设置 {name}；请在 {cfg_path.parent / '.env'} 中填写，或导出为环境变量"
            )
        return value.strip()

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    base = cfg_path.parent

    m = _section(raw, "models")
    models = ModelConfig(
        embedding_base_url=_require(m, "embedding_base_url", "models"),
        embedding_model=_require(m, "embedding_model", "models"),
        rerank_base_url=_require(m, "rerank_base_url", "models"),
        rerank_model=_require(m, "rerank_model", "models"),
        chat_base_url=_require(m, "chat_base_url", "models"),
        chat_model=_require(m, "chat_model", "models"),
        judge_model=_require(m, "judge_model", "models"),
        chat_temperature=float(_require(m, "chat_temperature", "models")),
        embedding_api_key=secret(EMBEDDING_KEY_ENV),
        chat_api_key=secret(CHAT_KEY_ENV),
        chat_extra_headers=_extra_headers(m),
    )

    c = _section(raw, "chunking")
    chunking = ChunkingConfig(
        child_target=int(_require(c, "child_target", "chunking")),
        child_max=int(_require(c, "child_max", "chunking")),
        parent_target=int(_require(c, "parent_target", "chunking")),
        parent_max=int(_require(c, "parent_max", "chunking")),
        context_max_chars=int(_require(c, "context_max_chars", "chunking")),
    )
    _check(
        chunking.child_target <= chunking.child_max,
        "配置 `chunking.child_target` 不能大于 `chunking.child_max`",
    )
    _check(
        chunking.parent_target <= chunking.parent_max,
        "配置 `chunking.parent_target` 不能大于 `chunking.parent_max`",
    )
    _check(
        chunking.child_max < chunking.parent_max,
        "配置 `chunking.child_max` 必须小于 `chunking.parent_max`，否则父块无法容纳子块",
    )

    ctx = _section(raw, "contextual")
    contextual = ContextualConfig(
        enabled=bool(ctx.get("enabled", True)),
        concurrency=int(ctx.get("concurrency", 4)),
        interval_seconds=float(ctx.get("interval_seconds", 0.0)),
    )
    _check(contextual.concurrency >= 1, "配置 `contextual.concurrency` 至少为 1")
    _check(contextual.interval_seconds >= 0, "配置 `contextual.interval_seconds` 不能为负")

    r = _section(raw, "retrieval")
    retrieval = RetrievalConfig(
        vec_top_k=int(_require(r, "vec_top_k", "retrieval")),
        bm25_top_k=int(_require(r, "bm25_top_k", "retrieval")),
        rrf_k=int(_require(r, "rrf_k", "retrieval")),
        rerank_top_n=int(_require(r, "rerank_top_n", "retrieval")),
        final_n=int(_require(r, "final_n", "retrieval")),
        gate_threshold=float(_require(r, "gate_threshold", "retrieval")),
    )
    _check(retrieval.vec_top_k > 0, "配置 `retrieval.vec_top_k` 必须为正")
    _check(retrieval.bm25_top_k > 0, "配置 `retrieval.bm25_top_k` 必须为正")
    _check(retrieval.rrf_k > 0, "配置 `retrieval.rrf_k` 必须为正")
    _check(retrieval.rerank_top_n > 0, "配置 `retrieval.rerank_top_n` 必须为正")
    _check(
        retrieval.final_n <= retrieval.rerank_top_n,
        "配置 `retrieval.final_n` 不能大于 `retrieval.rerank_top_n`",
    )

    w = raw.get("webui") or {}
    webui = WebUIConfig(
        server_name=str(w.get("server_name", "127.0.0.1")),
        server_port=int(w.get("server_port", 7860)),
    )
    _check(0 < webui.server_port < 65536, "配置 `webui.server_port` 不在合法端口范围")

    mq = raw.get("multi_query") or {}
    multi_query = MultiQueryConfig(
        enabled=bool(mq.get("enabled", True)),
        num_queries=int(mq.get("num_queries", 4)),
        temperature=float(mq.get("temperature", 0.2)),
    )
    _check(multi_query.num_queries >= 1, "配置 `multi_query.num_queries` 至少为 1")
    _check(
        0 <= multi_query.temperature <= 1,
        "配置 `multi_query.temperature` 应在 0~1 之间",
    )

    kb_dir = _resolve(_require(raw, "kb_dir", "kb_dir"), base, "kb_dir")
    _check(kb_dir.is_dir(), f"配置 `kb_dir` 指向的目录不存在：{kb_dir}")
    index_dir = _resolve(_require(raw, "index_dir", "index_dir"), base, "index_dir")

    return Config(
        kb_dir=kb_dir,
        index_dir=index_dir,
        models=models,
        chunking=chunking,
        contextual=contextual,
        retrieval=retrieval,
        webui=webui,
        multi_query=multi_query,
    )

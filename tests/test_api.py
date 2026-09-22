"""HTTP 边界测试：路由、参数校验、字段映射、密钥擦除与图片端点。

全部用替身替换 `api.ask`，不发起任何真实模型调用。
"""

import os
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from xiaolinrag import api
from xiaolinrag.config import (
    ChunkingConfig,
    Config,
    ContextualConfig,
    ModelConfig,
    MultiQueryConfig,
    RetrievalConfig,
    WebUIConfig,
)
from xiaolinrag.index_store import IndexEntry, IndexStore
from xiaolinrag.llm import LLMError
from xiaolinrag.pipeline import AnswerResult, Citation

EMBED_KEY = "sk-embed-secret"
CHAT_KEY = "sk-chat-secret"


def make_cfg() -> Config:
    return Config(
        kb_dir=Path("/tmp/kb"),
        index_dir=Path("/tmp/index"),
        models=ModelConfig(
            "u", "embed-model", "u", "rerank-model", "u", "chat-model", "judge", 0.3,
            EMBED_KEY, CHAT_KEY,
        ),
        chunking=ChunkingConfig(300, 500, 900, 1200, 100),
        contextual=ContextualConfig(True, 4, 0.0),
        retrieval=RetrievalConfig(20, 20, 60, 40, 5, 0.3),
        webui=WebUIConfig("127.0.0.1", 7860),
        multi_query=MultiQueryConfig(False, 4, 0.2),
    )


def entry(chunk_id: str, section: str = "xiaolinnote/rag") -> IndexEntry:
    return IndexEntry(
        chunk_id=chunk_id,
        parent_id=f"{chunk_id}#p0",
        page_id=chunk_id.split("#")[0],
        section=section,
        title=f"{chunk_id} 的标题",
        source_url=f"https://example.com/{chunk_id}",
        heading_path="示例 > 小节",
        child_text=f"{chunk_id} 的子块原文",
        context_note="",
        parent_text=f"{chunk_id} 的父块原文",
        seq=0,
    )


@pytest.fixture
def store() -> IndexStore:
    return IndexStore(
        entries=[
            entry("xiaolinnote/rag/a#c0"),
            entry("xiaolinnote/llm/b#c0", section="xiaolinnote/llm"),
            entry("xiaolincoding/network/c#c0", section="xiaolincoding/network"),
        ],
        index=None,  # ask 被替换掉，检索通道不会被触碰
        meta={
            "built_at": "2026-09-20T08:49:22+00:00",
            "embedding_model": "Qwen/Qwen3-Embedding-8B",
            "chat_model": "chat-model",
            "page_count": 3,
            "parent_count": 3,
            "site_labels": {"xiaolinnote": "小林面试笔记", "xiaolincoding": "小林coding"},
            "section_labels": {
                "xiaolinnote/rag": "RAG",
                "xiaolinnote/llm": "LLM",
                "xiaolincoding/network": "图解网络",
            },
        },
    )


@pytest.fixture
def client(store) -> TestClient:
    return TestClient(api.create_app(make_cfg(), store))


def result(rejected: bool = False) -> AnswerResult:
    return AnswerResult(
        rejected=rejected,
        answer="门控未通过" if rejected else "答案正文 [1]",
        citations=[] if rejected else [
            Citation(
                index=1,
                page_id="xiaolinnote/rag/a",
                title="分块策略",
                section="xiaolinnote/rag",
                section_label="RAG",
                source_url="https://example.com/rag/a#c0",
                heading_path="4. RAG > 粒度",
                excerpt="父块原文",
            )
        ],
        debug={"counts": {"vector": 2, "keyword": 1}, "gate_threshold": 0.3},
    )


def stub(monkeypatch, value, record: list | None = None):
    def fake_ask(question, cfg, store, *, sections=None, debug=False):
        if record is not None:
            record.append({"question": question, "sections": sections, "debug": debug})
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(api, "ask", fake_ask)


# ---------- 元信息 ----------

def test_meta_reports_counts_and_sections(client):
    data = client.get("/api/meta").json()

    assert data["page_count"] == 3
    assert data["child_count"] == 3
    assert data["parent_count"] == 3
    assert data["built_at"] == "2026-09-20T08:49:22+00:00"
    assert data["embedding_model"] == "Qwen/Qwen3-Embedding-8B"
    assert data["rerank_model"] == "rerank-model"
    assert data["chat_model"] == "chat-model"
    # 站点按索引里的声明顺序（不是字典序），栏目仍按 key 字典序
    assert data["sites"] == [
        {
            "key": "xiaolinnote",
            "label": "小林面试笔记",
            "sections": [
                {"key": "xiaolinnote/llm", "label": "LLM"},
                {"key": "xiaolinnote/rag", "label": "RAG"},
            ],
        },
        {
            "key": "xiaolincoding",
            "label": "小林coding",
            "sections": [{"key": "xiaolincoding/network", "label": "图解网络"}],
        },
    ]


def test_meta_carries_no_key_material(client):
    body = client.get("/api/meta").text

    assert EMBED_KEY not in body
    assert CHAT_KEY not in body


# ---------- 静态页面 ----------

def test_index_page_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "<title>小林面试笔记 RAG 问答</title>" in response.text


@pytest.mark.parametrize("path,marker", [("/styles.css", "--accent"), ("/app.js", "renderMarkdown")])
def test_static_assets_are_served(client, path, marker):
    response = client.get(path)

    assert response.status_code == 200
    assert marker in response.text


def test_unknown_path_is_404(client):
    assert client.get("/nonexistent").status_code == 404


# ---------- 问答 ----------

def test_ask_maps_answer_result_to_json(client, monkeypatch):
    stub(monkeypatch, result())

    data = client.post("/api/ask", json={"question": "粒度怎么定"}).json()

    assert data["rejected"] is False
    assert data["answer"] == "答案正文 [1]"
    assert data["debug"]["gate_threshold"] == 0.3
    citation = data["citations"][0]
    assert citation == {
        "index": 1,
        "page_id": "xiaolinnote/rag/a",
        "title": "分块策略",
        "section": "xiaolinnote/rag",
        "section_label": "RAG",
        "source_url": "https://example.com/rag/a#c0",
        "heading_path": "4. RAG > 粒度",
        "excerpt": "父块原文",
    }


def test_ask_forwards_sections_and_debug(client, monkeypatch):
    seen: list = []
    stub(monkeypatch, result(), seen)

    client.post("/api/ask", json={"question": "q", "sections": ["rag"], "debug": True})

    assert seen == [{"question": "q", "sections": ["rag"], "debug": True}]


def test_ask_empty_sections_means_whole_corpus(client, monkeypatch):
    seen: list = []
    stub(monkeypatch, result(), seen)

    client.post("/api/ask", json={"question": "q", "sections": []})

    assert seen[0]["sections"] is None


def test_ask_defaults_are_off(client, monkeypatch):
    seen: list = []
    stub(monkeypatch, result(), seen)

    client.post("/api/ask", json={"question": "q"})

    assert seen[0] == {"question": "q", "sections": None, "debug": False}


def test_ask_passes_rejection_through(client, monkeypatch):
    stub(monkeypatch, result(rejected=True))

    response = client.post("/api/ask", json={"question": "库外问题"})

    assert response.status_code == 200
    data = response.json()
    assert data["rejected"] is True
    assert data["answer"] == "门控未通过"
    assert data["citations"] == []


# ---------- 校验 ----------

def test_empty_question_is_readable_422(client):
    response = client.post("/api/ask", json={"question": ""})

    assert response.status_code == 422
    assert response.json()["detail"] == "问题不能为空"


def test_missing_question_is_readable_422(client):
    response = client.post("/api/ask", json={})

    assert response.status_code == 422
    assert response.json()["detail"] == "问题不能为空"


def test_overlong_question_is_readable_422(client):
    response = client.post("/api/ask", json={"question": "字" * (api.QUESTION_MAX + 1)})

    assert response.status_code == 422
    assert "过长" in response.json()["detail"]


# ---------- 错误与密钥 ----------

def test_ask_failure_returns_502_with_readable_detail(client, monkeypatch):
    stub(monkeypatch, LLMError("上游返回 503"))

    response = client.post("/api/ask", json={"question": "q"})

    assert response.status_code == 502
    assert response.json()["detail"] == "上游返回 503"


def test_ask_failure_redacts_keys_in_response(client, monkeypatch):
    stub(monkeypatch, LLMError(f"鉴权失败：Bearer {EMBED_KEY} 与 {CHAT_KEY}"))

    body = client.post("/api/ask", json={"question": "q"}).text

    assert EMBED_KEY not in body
    assert CHAT_KEY not in body
    assert body.count("***") == 2


def test_ask_failure_redacts_keys_in_logs(client, monkeypatch, caplog):
    stub(monkeypatch, LLMError(f"鉴权失败：Bearer {EMBED_KEY}"))

    with caplog.at_level("WARNING", logger="xiaolinrag.api"):
        client.post("/api/ask", json={"question": "q"})

    assert EMBED_KEY not in caplog.text
    assert "***" in caplog.text


# ---------- 图片端点 ----------

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def kb_root(tmp_path) -> Path:
    """迷你知识库，模拟真实的变深布局：一个两层站点，另一个站点更深一层。

    两张合法图片、一个 svg、一篇 markdown，另加一个指向库外的符号链接。
    """
    shallow = tmp_path / "xiaolinnote" / "rag" / "assets" / "demo"
    shallow.mkdir(parents=True)
    (shallow / "ok.png").write_bytes(PNG_MAGIC + b"0" * 32)
    (shallow / "pic.webp").write_bytes(b"RIFF" + b"0" * 32)
    (shallow / "icon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "xiaolinnote" / "rag" / "article.md").write_text("# 标题", encoding="utf-8")
    os.symlink("/etc/passwd", shallow / "escape.png")

    deep = tmp_path / "xiaolincoding" / "os" / "1_hardware" / "assets" / "how_cpu_run"
    deep.mkdir(parents=True)
    (deep / "deep.png").write_bytes(PNG_MAGIC + b"1" * 32)
    return tmp_path


@pytest.fixture
def asset_client(kb_root, store) -> TestClient:
    return TestClient(api.create_app(replace(make_cfg(), kb_dir=kb_root), store))


def test_asset_serves_png(asset_client):
    response = asset_client.get("/kb/xiaolinnote/rag/assets/demo/ok.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(PNG_MAGIC)


def test_asset_serves_webp_with_explicit_media_type(asset_client):
    """标准库的 mimetypes 不认识 .webp，靠显式映射兜住。"""
    response = asset_client.get("/kb/xiaolinnote/rag/assets/demo/pic.webp")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"


def test_asset_serves_image_under_deep_path(asset_client):
    """站点/栏目/章节/assets/文章/图片 —— 层级不固定，任意深度都要能取到。"""
    response = asset_client.get(
        "/kb/xiaolincoding/os/1_hardware/assets/how_cpu_run/deep.png"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


@pytest.mark.parametrize(
    "path",
    [
        "/kb/xiaolinnote/rag/article.md",  # 库内但不是图片
        "/kb/xiaolinnote/rag/assets/demo/icon.svg",  # 白名单外
        "/kb/xiaolinnote/rag/assets/demo/ok.txt",  # 白名单外
        "/kb/xiaolinnote/rag/assets/demo/missing.png",  # 不存在
        "/kb/assets/demo/ok.png",  # 不以 assets 目录为基准（路径里没有 <前缀>/assets/）
        "/kb/xiaolinnote/rag/assets/%2e%2e%2f%2e%2e%2f%2e%2e%2fetc/passwd.png",  # 编码穿越
    ],
)
def test_asset_requests_are_rejected(asset_client, path):
    response = asset_client.get(path)

    assert response.status_code == 404
    assert response.json()["detail"] == "资源不存在"


def test_asset_rejection_leaks_no_server_path(asset_client, kb_root):
    body = asset_client.get("/kb/xiaolinnote/rag/assets/demo/missing.png").text

    assert str(kb_root) not in body
    assert "/workspace" not in body


@pytest.mark.parametrize(
    "raw",
    [
        "../config.yaml",
        "xiaolinnote/../config.yaml",
        "xiaolinnote/rag/assets/../../../etc/passwd.png",
        "xiaolinnote/rag/assets/demo/../../../../etc/passwd.png",
        "xiaolinnote/rag/article.md",
        "xiaolinnote/rag/assets/demo/icon.svg",
        "xiaolinnote/rag/assets/demo/escape.png",  # 符号链接指向库外，resolve 后必须被挡下
        "assets/demo/ok.png",  # 不以 assets 目录为基准，不在形状内
        "/etc/passwd.png",  # 绝对路径
    ],
)
def test_resolve_asset_rejects_directly(kb_root, raw):
    """直接测解析函数：HTTP 层会先规范化 URL，只靠接口测不出这些穿越形态。"""
    cfg = replace(make_cfg(), kb_dir=kb_root)

    with pytest.raises(HTTPException) as caught:
        api._resolve_asset(cfg, raw)

    assert caught.value.status_code == 404
    assert caught.value.detail == "资源不存在"


def test_resolve_asset_accepts_valid_image(kb_root):
    cfg = replace(make_cfg(), kb_dir=kb_root)

    target, media_type = api._resolve_asset(cfg, "xiaolinnote/rag/assets/demo/ok.png")

    assert target.read_bytes().startswith(PNG_MAGIC)
    assert media_type == "image/png"

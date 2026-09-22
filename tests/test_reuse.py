from pathlib import Path

import numpy as np
import pytest

from xiaolinrag.config import (
    ChunkingConfig,
    Config,
    ContextualConfig,
    ModelConfig,
    RetrievalConfig,
    WebUIConfig,
)
from xiaolinrag.index_store import IndexEntry, IndexStore, build_meta
from xiaolinrag.reuse import ReuseSource

DIM = 4


def make_cfg(chat_model: str = "chat-x", embedding_model: str = "embed-x") -> Config:
    return Config(
        kb_dir=Path("/tmp/kb"),
        index_dir=Path("/tmp/index"),
        models=ModelConfig(
            "u", embedding_model, "u", "m", "u", chat_model, "m", 0.3, "sk-embed", "sk-chat"
        ),
        chunking=ChunkingConfig(300, 500, 900, 1200, 100),
        contextual=ContextualConfig(True, 4, 0.0),
        retrieval=RetrievalConfig(20, 20, 60, 40, 5, 0.3),
        webui=WebUIConfig("127.0.0.1", 7860),
    )


def entry(page_id: str, seq: int, child_text: str, context_note: str) -> IndexEntry:
    return IndexEntry(
        chunk_id=f"{page_id}#c{seq}",
        parent_id=f"{page_id}#p0",
        page_id=page_id,
        section=page_id.rsplit("/", 1)[0],
        title="示例标题",
        source_url="https://example.com",
        heading_path="示例 > 小节",
        child_text=child_text,
        context_note=context_note,
        parent_text=child_text,
        seq=seq,
    )


def write_index(tmp_path, entries, cfg: Config) -> Path:
    """把一份索引落盘，当作「上一版」。"""
    vectors = np.asarray(
        [[float(i + 1), 0.5, 0.25, 0.125] for i in range(len(entries))], dtype="float32"
    )
    meta = build_meta(
        embedding_model=cfg.models.embedding_model,
        chat_model=cfg.models.chat_model,
        page_count=1,
        parent_count=1,
        chunking=cfg.chunking,
        contextual=cfg.contextual,
    )
    IndexStore.build(entries, vectors, meta).save(tmp_path)
    return tmp_path


@pytest.fixture
def previous(tmp_path):
    """上一版索引：两篇文章，各两个子块。"""
    cfg = make_cfg()
    entries = [
        entry("xiaolinnote/rag/a", 0, "A 的第一个子块", "A 的背景一"),
        entry("xiaolinnote/rag/a", 1, "A 的第二个子块", "A 的背景二"),
        entry("xiaolinnote/llm/b", 0, "B 的第一个子块", "B 的背景一"),
        entry("xiaolinnote/llm/b", 1, "B 的第二个子块", "B 的背景二"),
    ]
    return write_index(tmp_path, entries, cfg), cfg


def test_same_content_hits_contexts_and_vectors(previous):
    index_dir, cfg = previous
    reuse = ReuseSource.load(index_dir, cfg)

    assert reuse.available
    key = reuse.context_key(["A 的第一个子块", "A 的第二个子块"], cfg)
    assert reuse.contexts(key) == ["A 的背景一", "A 的背景二"]

    vec_key = reuse.vector_key("A 的背景一\n\nA 的第一个子块")
    assert reuse.vector(vec_key) is not None
    assert reuse.stats.context_pages_hit == 1
    assert reuse.stats.vector_hit == 1


def test_changed_child_text_misses(previous):
    index_dir, cfg = previous
    reuse = ReuseSource.load(index_dir, cfg)

    assert reuse.contexts(reuse.context_key(["A 的第一个子块改过了"], cfg)) is None
    assert reuse.vector(reuse.vector_key("完全没见过的文本")) is None
    assert reuse.stats.context_pages_missed == 1
    assert reuse.stats.vector_missed == 1


def test_chunking_change_invalidates_contexts(previous):
    index_dir, cfg = previous
    reuse = ReuseSource.load(index_dir, cfg)
    key_before = reuse.context_key(["A 的第一个子块", "A 的第二个子块"], cfg)

    changed = make_cfg()
    changed.chunking = ChunkingConfig(400, 600, 900, 1200, 100)
    assert reuse.context_key(["A 的第一个子块", "A 的第二个子块"], changed) != key_before


def test_chat_model_change_disables_context_reuse(previous):
    index_dir, _ = previous
    reuse = ReuseSource.load(index_dir, make_cfg(chat_model="chat-y"))

    assert reuse.available, "索引本身读得进来，只是模型对不上"
    assert "对话模型" in reuse.reason
    key = reuse.context_key(["A 的第一个子块", "A 的第二个子块"], make_cfg(chat_model="chat-y"))
    assert reuse.contexts(key) is None


def test_embedding_model_change_disables_vector_reuse(previous):
    index_dir, _ = previous
    reuse = ReuseSource.load(index_dir, make_cfg(embedding_model="embed-y"))

    assert "向量模型" in reuse.reason
    assert reuse.vector(reuse.vector_key("A 的背景一\n\nA 的第一个子块")) is None


def test_title_change_does_not_affect_hit(tmp_path):
    """标题不进键：同一批子块换个标题仍应命中。"""
    cfg = make_cfg()
    before = [entry("xiaolinnote/rag/a", i, f"子块{i}", f"背景{i}") for i in range(2)]
    write_index(tmp_path, before, cfg)

    after = [entry("xiaolinnote/rag/a", i, f"子块{i}", f"背景{i}") for i in range(2)]
    for item in after:
        item.title = "换了标题"
    IndexStore.build(
        after,
        np.asarray([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype="float32"),
        build_meta(
            embedding_model=cfg.models.embedding_model,
            chat_model=cfg.models.chat_model,
            page_count=1,
            parent_count=1,
            chunking=cfg.chunking,
            contextual=cfg.contextual,
        ),
    ).save(tmp_path)

    reuse = ReuseSource.load(tmp_path, cfg)
    assert reuse.contexts(reuse.context_key(["子块0", "子块1"], cfg)) == ["背景0", "背景1"]


def test_entry_order_does_not_matter(tmp_path):
    """复用按内容查、不按下标查，所以上一版的作废顺序不影响命中。"""
    cfg = make_cfg()
    ordered = [
        entry("xiaolinnote/rag/a", 0, "A 的子块", "A 的背景"),
        entry("xiaolinnote/llm/b", 0, "B 的子块", "B 的背景"),
    ]
    write_index(tmp_path, ordered, cfg)

    reuse = ReuseSource.load(tmp_path, cfg)
    assert reuse.contexts(reuse.context_key(["B 的子块"], cfg)) == ["B 的背景"]
    assert reuse.contexts(reuse.context_key(["A 的子块"], cfg)) == ["A 的背景"]


def test_all_empty_notes_are_treated_as_miss(tmp_path):
    """整篇背景都为空说明时宁可重算，不把上一版的失败固化下来。"""
    cfg = make_cfg()
    write_index(tmp_path, [entry("xiaolinnote/rag/a", 0, "子块", "")], cfg)

    reuse = ReuseSource.load(tmp_path, cfg)
    assert reuse.contexts(reuse.context_key(["子块"], cfg)) is None


def test_missing_index_degrades_to_empty(tmp_path):
    reuse = ReuseSource.load(tmp_path / "nowhere", make_cfg())

    assert not reuse.available
    assert "没有上一版索引" in reuse.reason
    assert reuse.contexts("whatever") is None
    assert reuse.vector("whatever") is None


def test_corrupted_entries_degrades_to_empty(previous):
    index_dir, cfg = previous
    (index_dir / "entries.jsonl").write_text("{ 这不是 JSON\n", encoding="utf-8")

    reuse = ReuseSource.load(index_dir, cfg)

    assert not reuse.available
    assert "读不出来" in reuse.reason


def test_vector_count_mismatch_degrades(tmp_path):
    """向量数与条目数对不上时只关掉向量复用，不整个失效。"""
    cfg = make_cfg()
    write_index(tmp_path, [entry("xiaolinnote/rag/a", 0, "子块", "背景")], cfg)
    (tmp_path / "entries.jsonl").write_text(
        (tmp_path / "entries.jsonl").read_text(encoding="utf-8")
        + (tmp_path / "entries.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    reuse = ReuseSource.load(tmp_path, cfg)

    assert reuse.vector(reuse.vector_key("背景\n\n子块")) is None

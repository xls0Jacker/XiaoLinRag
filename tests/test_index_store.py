import json

import numpy as np
import pytest

from xiaolinrag.index_store import IndexEntry, IndexStore, IndexStoreError

META = {"embedding_model": "test-model", "page_count": 2, "parent_count": 2}


def make_entry(seq: int, section: str, text: str) -> IndexEntry:
    return IndexEntry(
        chunk_id=f"{section}/demo#c{seq}",
        parent_id=f"{section}/demo#p0",
        page_id=f"{section}/demo",
        section=section,
        title=f"{section} 示例",
        source_url=f"https://example.com/{section}/demo.html",
        heading_path="示例 > 小节",
        child_text=text,
        context_note="",
        parent_text=f"{text}（含前后文）",
        seq=seq,
    )


ENTRIES = [
    make_entry(0, "rag", "多路召回用 RRF 融合两路结果"),
    make_entry(1, "rag", "父子切割兼顾检索精度与上下文完整"),
    make_entry(2, "llm", "KV cache 减少重复计算"),
    make_entry(3, "llm", "注意力机制捕捉长距离依赖"),
]
VECTORS = np.eye(4, dtype=np.float32)


@pytest.fixture
def store():
    return IndexStore.build(ENTRIES, VECTORS, dict(META))


def test_build_aligns_rows(store):
    assert store.index.ntotal == len(ENTRIES)
    assert store.meta["dim"] == 4
    assert store.meta["child_count"] == 4


def test_build_rejects_count_mismatch():
    with pytest.raises(IndexStoreError, match="不一致"):
        IndexStore.build(ENTRIES, VECTORS[:2], dict(META))


def test_build_rejects_empty():
    with pytest.raises(IndexStoreError, match="没有可索引"):
        IndexStore.build([], np.zeros((0, 4), dtype=np.float32), dict(META))


def test_save_load_roundtrip(store, tmp_path):
    store.save(tmp_path)
    assert IndexStore.exists(tmp_path)

    loaded = IndexStore.load(tmp_path)

    assert [e.chunk_id for e in loaded.entries] == [e.chunk_id for e in ENTRIES]
    assert loaded.index.ntotal == len(ENTRIES)
    assert loaded.meta["embedding_model"] == "test-model"


def test_load_requires_build_first(tmp_path):
    with pytest.raises(IndexStoreError, match="请先运行"):
        IndexStore.load(tmp_path)


def test_load_detects_row_mismatch(store, tmp_path):
    store.save(tmp_path)
    lines = (tmp_path / "entries.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "entries.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(IndexStoreError, match="请重新运行"):
        IndexStore.load(tmp_path)


def test_sections_property(store):
    assert store.sections == ["llm", "rag"]


def test_search_vector_ranks_nearest(store):
    hits = store.search_vector(np.array([0, 0, 1, 0], dtype=np.float32), top_k=2)

    assert hits[0].entry.chunk_id == "llm/demo#c2"
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].rerank_score is None


def test_search_vector_section_filter(store):
    hits = store.search_vector(np.array([1, 0, 0, 0], dtype=np.float32), top_k=4, sections=["llm"])

    assert {h.entry.section for h in hits} == {"llm"}
    assert len(hits) == 2


def test_search_vector_unknown_section_returns_empty(store):
    hits = store.search_vector(np.array([1, 0, 0, 0], dtype=np.float32), top_k=4, sections=["nope"])

    assert hits == []


def test_search_vector_dim_mismatch(store):
    with pytest.raises(IndexStoreError, match="维度"):
        store.search_vector(np.zeros(3, dtype=np.float32), top_k=1)


def test_search_bm25_hits_keyword(store):
    hits = store.search_bm25("KV cache", top_k=2)

    assert hits, "BM25 没有召回任何结果"
    assert hits[0].entry.chunk_id == "llm/demo#c2"


def test_search_bm25_section_filter(store):
    hits = store.search_bm25("切割", top_k=4, sections=["llm"])

    assert hits == [], "栏目过滤应排除 rag 栏目里的命中"


def test_search_bm25_empty_query(store):
    assert store.search_bm25("？！", top_k=4) == []


def test_embed_text_includes_context_note():
    entry = make_entry(0, "rag", "子块原文")
    entry.context_note = "这段话在讲多路召回"

    assert entry.embed_text == "这段话在讲多路召回\n\n子块原文"


def test_entries_jsonl_is_readable(store, tmp_path):
    store.save(tmp_path)
    first = json.loads((tmp_path / "entries.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert first["chunk_id"] == "rag/demo#c0"
    assert first["parent_text"].endswith("（含前后文）")

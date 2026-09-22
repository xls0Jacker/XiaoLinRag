from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np
import pytest

from xiaolinrag.config import (
    ChunkingConfig,
    Config,
    ContextualConfig,
    ModelConfig,
    MultiQueryConfig,
    RetrievalConfig,
    WebUIConfig,
)
from xiaolinrag.llm import LLMError
from xiaolinrag.index_store import IndexEntry, ScoredHit
from xiaolinrag.pipeline import ask


def make_cfg(
    gate: float = 0.5,
    final_n: int = 5,
    rerank_top_n: int = 40,
    multi_query: MultiQueryConfig = MultiQueryConfig(False, 4, 0.2),
) -> Config:
    return Config(
        kb_dir=Path("/tmp/kb"),
        index_dir=Path("/tmp/index"),
        models=ModelConfig(
            "u", "m", "u", "m", "u", "m", "m", 0.3, "sk-embed", "sk-chat"
        ),
        chunking=ChunkingConfig(300, 500, 900, 1200, 100),
        contextual=ContextualConfig(True, 4, 0.0),
        retrieval=RetrievalConfig(20, 20, 60, rerank_top_n, final_n, gate),
        webui=WebUIConfig("127.0.0.1", 7860),
        multi_query=multi_query,
    )


def entry(chunk_id: str, parent_id: str, section: str = "xiaolinnote/rag") -> IndexEntry:
    return IndexEntry(
        chunk_id=chunk_id,
        parent_id=parent_id,
        page_id=parent_id.split("#")[0],
        section=section,
        title=f"{parent_id} 的标题",
        source_url=f"https://example.com/{parent_id}",
        heading_path="示例 > 小节",
        child_text=f"{chunk_id} 的子块原文",
        context_note="",
        parent_text=f"{parent_id} 的父块原文",
        seq=0,
    )


SECTION_LABELS = {"xiaolinnote/rag": "RAG", "xiaolinnote/llm": "LLM"}


class FakeStore:
    def __init__(self, vector_hits, keyword_hits, meta=None):
        self._vector_hits = vector_hits
        self._keyword_hits = keyword_hits
        self.meta = {"section_labels": SECTION_LABELS} if meta is None else meta
        self.calls = []

    def search_vector(self, vector, top_k, sections=None):
        self.calls.append(("vector", top_k, sections))
        return self._vector_hits

    def search_bm25(self, query, top_k, sections=None):
        self.calls.append(("bm25", top_k, sections))
        return self._keyword_hits


class FakeReranker:
    def __init__(self, scored):
        self._scored = scored
        self.seen = []

    def rerank(self, query, entries, cfg, top_n=None):
        self.seen.append(list(entries))
        return [
            ScoredHit(entry=e, score=s, rerank_score=s) for e, s in self._scored
        ]


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts = []

    def chat(self, system, user, cfg):
        self.prompts.append(user)
        return self.reply


@pytest.fixture
def embedder():
    return SimpleNamespace(embed_query=lambda query, cfg: np.zeros(4, dtype=np.float32))


def test_gate_rejects_low_score(embedder):
    target = entry("rag/a#c0", "rag/a#p0")
    store = FakeStore([ScoredHit(target, 0.9)], [])
    reranker = FakeReranker([(target, 0.05)])
    llm = FakeLLM("不该被调用")

    result = ask("红烧肉怎么做", make_cfg(gate=0.5), store, embedder, reranker, llm)

    assert result.rejected is True
    assert "未找到相关内容" in result.answer
    assert result.citations == []
    assert llm.prompts == [], "门控拒答时不应调用生成模型"


def test_gate_passes_and_generates(embedder):
    target = entry("rag/a#c0", "rag/a#p0")
    store = FakeStore([ScoredHit(target, 0.9)], [ScoredHit(target, 3.2)])
    reranker = FakeReranker([(target, 0.93)])
    llm = FakeLLM("多路召回就是同时用多种检索方式捞候选。[1]")

    result = ask("什么是多路召回", make_cfg(gate=0.5), store, embedder, reranker, llm)

    assert result.rejected is False
    assert result.answer.startswith("多路召回就是")
    assert [c.index for c in result.citations] == [1]
    assert result.citations[0].source_url == "https://example.com/rag/a#p0"
    assert result.citations[0].section_label == "RAG"


def test_citation_carries_page_id(embedder):
    """引用要带上文章标识，界面才能定位图片所在的目录。"""
    target = entry("rag/a#c0", "rag/a#p0")
    store = FakeStore([ScoredHit(target, 0.9)], [])

    result = ask(
        "什么是多路召回",
        make_cfg(gate=0.5),
        store,
        embedder,
        FakeReranker([(target, 0.93)]),
        FakeLLM("答案 [1]"),
    )

    assert result.citations[0].page_id == "rag/a"


def test_section_label_falls_back_to_last_segment(embedder):
    """索引元信息里没有展示名时兜底取复合键末段，而不是留空。"""
    target = entry("net/a#c0", "net/a#p0", section="xiaolincoding/network")
    store = FakeStore([ScoredHit(target, 0.9)], [], meta={})

    result = ask(
        "TCP 三次握手",
        make_cfg(gate=0.5),
        store,
        embedder,
        FakeReranker([(target, 0.93)]),
        FakeLLM("答案 [1]"),
    )

    assert result.citations[0].section_label == "network"


def test_prompt_uses_section_display_name(embedder):
    """提示词里的出处行写展示名，不写复合键。"""
    target = entry("rag/a#c0", "rag/a#p0")
    store = FakeStore([ScoredHit(target, 0.9)], [])
    llm = FakeLLM("答案 [1]")

    ask("什么是多路召回", make_cfg(gate=0.5), store, embedder, FakeReranker([(target, 0.93)]), llm)

    assert "（RAG ·" in llm.prompts[0]
    assert "xiaolinnote/rag" not in llm.prompts[0]


def test_empty_retrieval_is_rejected(embedder):
    store = FakeStore([], [])
    reranker = FakeReranker([])
    llm = FakeLLM("不该被调用")

    result = ask("无关问题", make_cfg(), store, embedder, reranker, llm)

    assert result.rejected is True
    assert "没有召回任何候选片段" in result.answer
    assert llm.prompts == []


def test_sections_forwarded_to_both_channels(embedder):
    target = entry("llm/a#c0", "llm/a#p0", section="xiaolinnote/llm")
    store = FakeStore([ScoredHit(target, 0.8)], [ScoredHit(target, 1.5)])
    reranker = FakeReranker([(target, 0.9)])

    ask(
        "注意力机制",
        make_cfg(),
        store,
        embedder,
        reranker,
        FakeLLM("答案 [1]"),
        sections=["xiaolinnote/llm"],
    )

    assert store.calls == [
        ("vector", 20, ["xiaolinnote/llm"]),
        ("bm25", 20, ["xiaolinnote/llm"]),
    ]


def test_parent_deduplication(embedder):
    first = entry("rag/a#c0", "rag/a#p0")
    second = entry("rag/a#c1", "rag/a#p0")  # 与 first 同父块
    third = entry("rag/b#c0", "rag/b#p0")
    store = FakeStore([ScoredHit(first, 0.9)], [])
    reranker = FakeReranker([(first, 0.95), (second, 0.9), (third, 0.85)])
    llm = FakeLLM("答案 [1] [2]")

    result = ask("问题", make_cfg(gate=0.5), store, embedder, reranker, llm)

    assert len(result.debug["selected"]) == 2, "同一父块的两个子块应合并为一条资料"
    assert llm.prompts[0].count("的父块原文") == 2
    assert [c.index for c in result.citations] == [1, 2]


def test_final_n_limits_sources(embedder):
    pool = [entry(f"rag/a#c{i}", f"rag/a#p{i}") for i in range(8)]
    store = FakeStore([ScoredHit(e, 0.9) for e in pool], [])
    reranker = FakeReranker([(e, 0.9 - i * 0.01) for i, e in enumerate(pool)])

    result = ask("问题", make_cfg(gate=0.5, final_n=3), store, embedder, reranker, FakeLLM("答案"))

    assert len(result.debug["selected"]) == 3


def test_rerank_top_n_limits_candidates(embedder):
    pool = [entry(f"rag/a#c{i}", f"rag/a#p{i}") for i in range(10)]
    store = FakeStore([ScoredHit(e, 0.9) for e in pool], [])
    reranker = FakeReranker([(e, 0.9) for e in pool[:4]])

    ask("问题", make_cfg(gate=0.5, rerank_top_n=4), store, embedder, reranker, FakeLLM("答案"))

    assert len(reranker.seen[0]) == 4


def test_citation_order_and_deduplication(embedder):
    first = entry("rag/a#c0", "rag/a#p0")
    second = entry("rag/b#c0", "rag/b#p0")
    store = FakeStore([ScoredHit(first, 0.9)], [])
    reranker = FakeReranker([(first, 0.95), (second, 0.9)])
    llm = FakeLLM("先看 [2]，再看 [1]，然后又提到 [2]。")

    result = ask("问题", make_cfg(gate=0.5), store, embedder, reranker, llm)

    assert [c.index for c in result.citations] == [2, 1]
    assert result.debug["dangling_citations"] == []


def test_dangling_citation_is_recorded(embedder):
    target = entry("rag/a#c0", "rag/a#p0")
    store = FakeStore([ScoredHit(target, 0.9)], [])
    reranker = FakeReranker([(target, 0.95)])
    llm = FakeLLM("答案引用了不存在的 [7]。")

    result = ask("问题", make_cfg(gate=0.5), store, embedder, reranker, llm)

    assert result.citations == []
    assert result.debug["dangling_citations"] == [7]


def test_debug_detail_toggle(embedder):
    target = entry("rag/a#c0", "rag/a#p0")
    store = FakeStore([ScoredHit(target, 0.9)], [ScoredHit(target, 2.0)])
    reranker = FakeReranker([(target, 0.95)])

    brief = ask("问题", make_cfg(gate=0.5), store, embedder, reranker, FakeLLM("答案 [1]"))
    full = ask(
        "问题", make_cfg(gate=0.5), store, embedder, reranker, FakeLLM("答案 [1]"), debug=True
    )

    assert "rerank_scores" not in brief.debug
    assert "rerank_scores" in full.debug
    assert full.debug["vector_hits"] == [("rag/a#c0", 0.9)]
    assert full.debug["counts"]["fused"] == 1


def test_empty_question_rejected(embedder):
    with pytest.raises(ValueError, match="问题不能为空"):
        ask("   ", make_cfg(), FakeStore([], []), embedder, FakeReranker([]), FakeLLM("x"))


# ---------- 多 Query 展开（T4：多路精排、降级、开关） ----------


class ExpandingFakeLLM:
    """能区分「展开 prompt」与「生成 prompt」的假 LLM。

    展开返回 JSON 数组；生成返回固定答案。chat 需透传 temperature，否则展开调用直接报错。
    """

    def __init__(self, expansion: list[str] | None, reply: str = "答案 [1]", *, error: bool = False):
        self.expansion = expansion
        self.reply = reply
        self.error = error
        self.expand_calls = 0
        self.gen_prompts = []

    def chat(self, system, user, cfg, **kwargs):
        if "你负责把用户的问题拆解成" in system:
            self.expand_calls += 1
            if self.error:
                raise LLMError("展开模型不可用")
            return json.dumps(self.expansion, ensure_ascii=False)
        self.gen_prompts.append(user)
        return self.reply


class MultiFakeReranker:
    """每个 query 播一段内定打分，记录收到的 query 与候选，供断言聚合行为。"""

    def __init__(self, per_query: dict):
        self.per_query = per_query  # {query: {chunk_id: score}}
        self.calls: list[tuple[str, list, int | None]] = []

    def rerank(self, query, entries, cfg, top_n=None):
        self.calls.append((query, list(entries), top_n))
        scores = self.per_query.get(query, {})
        scored = [
            ScoredHit(entry=e, score=scores.get(e.chunk_id, 0.0), rerank_score=scores.get(e.chunk_id, 0.0))
            for e in entries
        ]
        scored.sort(key=lambda h: (-h.rerank_score, h.entry.chunk_id))
        return scored


def make_multi_entry(chunk_id: str) -> IndexEntry:
    return entry(chunk_id, chunk_id.split("#")[0] + "#p0")


def test_multi_rerank_aggregates_max_per_query(embedder):
    """多路精排：用各展开 query 打分并按子块取 max 聚合，原复合 query 不参与打分。"""
    a = make_multi_entry("rag/a#c0")
    b = make_multi_entry("rag/b#c0")
    store = FakeStore([ScoredHit(a, 0.9), ScoredHit(b, 0.8)], [])
    original_q = "三次握手和四次挥手分别发生在什么时候"
    reranker = MultiFakeReranker(
        {
            "e1": {"rag/a#c0": 0.90, "rag/b#c0": 0.10},  # e1 是握手视角
            "e2": {"rag/b#c0": 0.95, "rag/a#c0": 0.20},  # e2 是挥手视角
        }
    )
    llm = ExpandingFakeLLM(["e1", "e2"])

    result = ask(
        original_q,
        make_cfg(gate=0.3, multi_query=MultiQueryConfig(True, 3, 0.2)),
        store,
        embedder,
        reranker,
        llm,
        debug=True,
    )

    # 召回/排序走展开聚合：精排只收到各聚焦 query（e1、e2），原复合 query 不参与打分
    assert [c[0] for c in reranker.calls] == ["e1", "e2"]
    # 聚合=max：e2 给 b 0.95 → 排前；a 只有 e1 的 0.90
    assert result.debug["rerank_scores"] == [("rag/b#c0", 0.95), ("rag/a#c0", 0.90)]
    assert result.debug["counts"]["selected"] == 2
    assert result.debug["queries"] == [original_q, "e1", "e2"]
    assert result.debug["expanded"] == ["e1", "e2"]
    assert len(result.debug["per_query"]) == 3


def test_multi_single_route_failure_skips_only_that_route(embedder):
    """任一展开 query 检索失败，只丢该路，其它路照常工作（N4）。"""
    a = make_multi_entry("rag/a#c0")
    store = FakeStore([ScoredHit(a, 0.9)], [])

    def boom_embedder(query, cfg):
        if query == "坏路":
            raise RuntimeError("向量模型崩了")
        return np.zeros(4, dtype=np.float32)

    reranker = MultiFakeReranker({"问题": {"rag/a#c0": 0.6}, "e1": {"rag/a#c0": 0.9}})
    llm = ExpandingFakeLLM(["坏路", "e1"])

    result = ask(
        "问题", make_cfg(gate=0.3, multi_query=MultiQueryConfig(True, 3, 0.2)),
        store, SimpleNamespace(embed_query=boom_embedder), reranker, llm,
    )

    assert result.rejected is False
    assert result.debug["counts"]["selected"] == 1
    assert "坏路" not in [c[0] for c in reranker.calls]


def test_multi_gate_rejects_when_aggregate_top_below_threshold(embedder):
    """多路模式下门控仍看聚合后的最高分，分数不足就拒答。"""
    a = make_multi_entry("rag/a#c0")
    store = FakeStore([ScoredHit(a, 0.9)], [])
    reranker = MultiFakeReranker({"e1": {"rag/a#c0": 0.20}})
    llm = ExpandingFakeLLM(["e1"])

    result = ask(
        "问题", make_cfg(gate=0.5, multi_query=MultiQueryConfig(True, 2, 0.2)),
        store, embedder, reranker, llm,
    )

    assert result.rejected is True
    assert "低于门控阈值" in result.answer
    assert llm.gen_prompts == []


def test_multi_degrades_to_original_when_expansion_fails(embedder):
    """LLM 展开失败时降级为原 query 单路检索，行为与旧链路一致（F6）。"""
    a = make_multi_entry("rag/a#c0")
    store = FakeStore([ScoredHit(a, 0.9)], [ScoredHit(a, 3.0)])
    reranker = MultiFakeReranker({"问题": {"rag/a#c0": 0.93}})
    llm = ExpandingFakeLLM(None, error=True)

    result = ask(
        "问题", make_cfg(gate=0.5, multi_query=MultiQueryConfig(True, 3, 0.2)),
        store, embedder, reranker, llm,
    )

    assert result.rejected is False
    assert [c[0] for c in reranker.calls] == ["问题"]
    assert "queries" not in result.debug  # 单路模式不输出多路键


def test_multi_disabled_does_not_call_llm(embedder):
    """开关关闭时：不调展开、单路精排、debug 无多路键（回归护栏）。"""
    a = make_multi_entry("rag/a#c0")
    store = FakeStore([ScoredHit(a, 0.9)], [ScoredHit(a, 2.0)])
    reranker = MultiFakeReranker({"问题": {"rag/a#c0": 0.9}})
    llm = ExpandingFakeLLM(["e1", "e2"])

    result = ask(
        "问题", make_cfg(gate=0.5, multi_query=MultiQueryConfig(False, 4, 0.2)),
        store, embedder, reranker, llm,
    )

    assert [c[0] for c in reranker.calls] == ["问题"]
    assert llm.expand_calls == 0
    assert "queries" not in result.debug
    assert result.debug["counts"]["selected"] == 1

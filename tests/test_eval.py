"""评估口径测试：检索层走共享 retrieve()（全库、无栏目过滤）、库外拒答断言。

全部用替身替换真实模型调用；skip_judge=True 绕开生成层裁判。
"""

import json
from pathlib import Path
from types import SimpleNamespace

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
from xiaolinrag.eval import EvalError, evaluate_retrieval, format_report, run_eval
from xiaolinrag.index_store import IndexEntry, ScoredHit


def make_cfg() -> Config:
    return Config(
        kb_dir=Path("/tmp/kb"),
        index_dir=Path("/tmp/index"),
        models=ModelConfig(
            "u", "m", "u", "m", "u", "m", "m", 0.3, "sk-embed", "sk-chat"
        ),
        chunking=ChunkingConfig(300, 500, 900, 1200, 100),
        contextual=ContextualConfig(True, 4, 0.0),
        retrieval=RetrievalConfig(20, 20, 60, 40, 5, 0.3),
        webui=WebUIConfig("127.0.0.1", 7860),
        multi_query=MultiQueryConfig(False, 4, 0.2),  # 单路，聚焦评估口径本身
    )


def entry(chunk_id: str, page_id: str) -> IndexEntry:
    return IndexEntry(
        chunk_id=chunk_id,
        parent_id=f"{chunk_id}#p0",
        page_id=page_id,
        section="rag",
        title=f"{chunk_id} 的标题",
        source_url=f"https://example.com/{chunk_id}",
        heading_path="示例 > 小节",
        child_text=f"{chunk_id} 的子块原文",
        context_note="",
        parent_text=f"{chunk_id} 的父块原文",
        seq=0,
    )


class FakeStore:
    """按 query 播内定命中；embedder 先把当前 query 记到 store 上。

    向量搜索拿不到 query 字样，只能靠 embedder 侧记。
    """

    def __init__(self, table: dict, meta=None):
        self._table = table  # {question: {"vector": [...], "keyword": [...]}}
        self.meta = {"section_labels": {}} if meta is None else meta
        self.sections_seen: list = []
        self.cur_query: str | None = None

    def search_vector(self, vector, top_k, sections=None):
        self.sections_seen.append(sections)
        return (self._table.get(self.cur_query, {}).get("vector") or [])[:top_k]

    def search_bm25(self, query, top_k, sections=None):
        self.sections_seen.append(sections)
        return (self._table.get(query, {}).get("keyword") or [])[:top_k]


class ScoredReranker:
    """按 chunk_id 给每个候选一个内定重排分。"""

    def __init__(self, scores: dict):
        self.scores = scores  # {chunk_id: score}

    def rerank(self, query, entries, cfg, top_n=None):
        pool = entries[:top_n] if top_n else entries
        scored = [
            ScoredHit(
                entry=e,
                score=self.scores.get(e.chunk_id, 0.0),
                rerank_score=self.scores.get(e.chunk_id, 0.0),
            )
            for e in pool
        ]
        scored.sort(key=lambda h: (-h.rerank_score, h.entry.chunk_id))
        return scored


class FakeLLM:
    def __init__(self, reply: str = "答案 [1]"):
        self.reply = reply
        self.calls = 0

    def chat(self, system, user, cfg, **kwargs):
        self.calls += 1
        return self.reply


def make_store(table: dict) -> FakeStore:
    store = FakeStore(table)
    return store


def make_embedder(store: FakeStore):
    def embed_query(query, cfg):
        store.cur_query = query
        return np.zeros(4, dtype=np.float32)

    return SimpleNamespace(embed_query=embed_query)


def write_set(tmp_path: Path, lines: list[dict]) -> Path:
    target = tmp_path / "set.jsonl"
    target.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
    return target


def test_retrieval_uses_whole_corpus_sections_none(tmp_path):
    """检索层切到共享 retrieve()：sections 恒 None（全库口径），不按栏目预过滤。"""
    target = entry("rag/a#c0", "p1")
    result = entry("rag/other#c0", "p999")
    store = make_store(
        {
            "什么是多路召回": {"vector": [ScoredHit(target, 0.9)], "keyword": []},
            "无关题": {"vector": [ScoredHit(result, 0.9)], "keyword": []},
        }
    )
    items = evaluate_retrieval(
        [
            {"id": "q1", "section": "rag", "page_id": "p1", "question": "什么是多路召回"},
            {"id": "q2", "section": "os", "page_id": "p999", "question": "无关题"},
        ],
        make_cfg(),
        store,
        make_embedder(store),
        ScoredReranker({"rag/a#c0": 0.9, "rag/other#c0": 0.9}),
        FakeLLM(),
    )

    assert [i.item_id for i in items] == ["q1", "q2"]
    assert all(section is None for section in store.sections_seen), "全库口径：不得传栏目过滤"
    assert items[0].hit and items[0].rank == 1


def test_expect_rejected_items_skip_hit_stats(tmp_path):
    """标 expect_rejected 的库外样本不进命中统计，也不该产生 EvalItem。"""
    target = entry("rag/a#c0", "p1")
    store = make_store(
        {
            "什么是多路召回": {"vector": [ScoredHit(target, 0.9)], "keyword": []},
            "InnoDB 和 MyISAM": {"vector": [ScoredHit(target, 0.9)], "keyword": []},  # 库里无关，假设召回了
        }
    )
    items = evaluate_retrieval(
        [
            {"id": "q1", "section": "rag", "page_id": "p1", "question": "什么是多路召回"},
            {"id": "out-001", "section": "db", "page_id": "out-001", "question": "InnoDB 和 MyISAM", "expect_rejected": True},
        ],
        make_cfg(),
        store,
        make_embedder(store),
        ScoredReranker({"rag/a#c0": 0.9}),
        FakeLLM(),
    )

    assert len(items) == 1
    assert "out-001" not in [i.item_id for i in items]


def test_run_eval_raises_when_out_of_kb_passes(tmp_path):
    """库外样本标了期望拒答却被门控放行 → run_eval 抛 EvalError 并列出 id。"""
    target = entry("rag/a#c0", "p1")
    store = make_store(
        {
            "什么是多路召回": {"vector": [ScoredHit(target, 0.9)], "keyword": []},
            "InnoDB 引擎原理": {"vector": [ScoredHit(target, 0.9)], "keyword": []},  # 不该有高命中，但假设放行
        }
    )
    set_path = write_set(
        tmp_path,
        [
            {"id": "q1", "section": "rag", "page_id": "p1", "question": "什么是多路召回"},
            {"id": "out-001", "section": "db", "page_id": "out-001", "question": "InnoDB 引擎原理", "expect_rejected": True},
        ],
    )

    with pytest.raises(EvalError, match="out-001"):
        run_eval(
            make_cfg(),
            store,
            set_path=set_path,
            skip_judge=True,
            embedder=make_embedder(store),
            reranker=ScoredReranker({"rag/a#c0": 0.9}),
            llm=FakeLLM(),
        )


def test_run_eval_passes_when_out_of_kb_rejected(tmp_path):
    """库外样本被门控拒答 → 不抛错，报告带断言段。"""
    target = entry("rag/a#c0", "p1")
    store = make_store(
        {
            "什么是多路召回": {"vector": [ScoredHit(target, 0.9)], "keyword": []},
            "InnoDB 引擎原理": {"vector": [], "keyword": []},  # 查不到 → 空召回 → 拒答
        }
    )
    set_path = write_set(
        tmp_path,
        [
            {"id": "q1", "section": "rag", "page_id": "p1", "question": "什么是多路召回"},
            {"id": "out-001", "section": "db", "page_id": "out-001", "question": "InnoDB 引擎原理", "expect_rejected": True},
        ],
    )

    report = run_eval(
        make_cfg(),
        store,
        set_path=set_path,
        skip_judge=True,
        embedder=make_embedder(store),
        reranker=ScoredReranker({"rag/a#c0": 0.9}),
        llm=FakeLLM(),
    )

    assert len(report.reject_checks) == 1
    assert report.reject_checks[0].rejected is True
    assert report.reject_checks[0].top_score is None
    text = format_report(report)
    assert "库外拒答断言" in text
    assert "[拒答] out-001" in text


def test_format_report_skips_when_no_out_of_kb(tmp_path):
    set_path = write_set(
        tmp_path,
        [
            {"id": "q1", "section": "rag", "page_id": "p1", "question": "什么是多路召回"},
        ],
    )
    target = entry("rag/a#c0", "p1")
    store = make_store({"什么是多路召回": {"vector": [ScoredHit(target, 0.9)], "keyword": []}})

    report = run_eval(
        make_cfg(),
        store,
        set_path=set_path,
        skip_judge=True,
        embedder=make_embedder(store),
        reranker=ScoredReranker({"rag/a#c0": 0.9}),
        llm=FakeLLM(),
    )

    assert "无库外样本（跳过断言）" in format_report(report)
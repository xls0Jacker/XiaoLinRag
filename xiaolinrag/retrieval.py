"""RRF 融合：把多路召回结果按倒数排名合并（对齐 rag/11、13 篇的方案）。

用排名而非原始分数融合，回避了向量相似度与 BM25 分数不可比的问题。
`fuse_rrf_multi` 把「双路」泛化到「多 query × 双通道」任意多路。
"""

from __future__ import annotations

from .index_store import IndexEntry, ScoredHit

DEFAULT_K = 60


def fuse_rrf_multi(
    hit_lists: list[list[ScoredHit]],
    k: int = DEFAULT_K,
) -> list[ScoredHit]:
    """跨任意多路再累计的倒数排名融合：每路按 Σ 1/(k+rank) 计分，同一子块多路命中则累加。

    返回按融合分降序排列的候选（已按 chunk_id 去重，每个子块只保留一个条目）；
    同分时按 chunk_id 排序，保证相同输入下结果稳定可复现（N3）。
    """
    scores: dict[str, float] = {}
    entries: dict[str, IndexEntry] = {}

    for hits in hit_lists:
        for rank, hit in enumerate(hits, start=1):
            key = hit.entry.chunk_id
            entries[key] = hit.entry
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [ScoredHit(entry=entries[key], score=score) for key, score in ordered]


def fuse_rrf(
    hits_vector: list[ScoredHit],
    hits_keyword: list[ScoredHit],
    k: int = DEFAULT_K,
) -> list[ScoredHit]:
    """双路兼容包装，委托给 `fuse_rrf_multi`；既有调用方与测试零改动。"""
    return fuse_rrf_multi([hits_vector, hits_keyword], k)

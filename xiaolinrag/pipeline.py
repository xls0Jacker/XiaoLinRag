"""在线查询编排：向量化 → 多路召回 → RRF 融合 → 精排 → 质量门控 → 生成 → 引用解析。

`retrieve()` 是「多 Query 展开 → 多路召回 → RRF 融合 → 精排」的共享编排（spec F9/plan.md），
在线 `ask` 与评估检索层共用同一函数，两处口径不会漂移。多 Query 展开对齐
kb/xiaolinnote/rag docs 12 方法四 / 13 第三路 / 14 Multi-Query：原 query 恒保留做召回，
精排改用各聚焦展开 query 打分后按块取 max 聚合（F4）——这是修复「复合 query 注水、
广告块顶榜、相关块垫底」的必要行为，不是可选项。

质量门控对齐 rag/17_hallucination.md：精排最高分低于阈值时直接拒答，
不让生成模型在低质量上下文上硬撑。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from . import embedder as embedder_api
from . import llm as llm_api
from . import reranker as reranker_api
from .config import Config
from .index_store import IndexEntry, IndexStore, ScoredHit
from .multi_query import expand_queries
from .prompts import SYSTEM_PROMPT, build_user_prompt, section_label
from .retrieval import fuse_rrf_multi

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d+)\]")

EMPTY_RETRIEVAL_MESSAGE = "知识库中未找到相关内容：这次检索没有召回任何候选片段，可能是问题超出了笔记覆盖的范围。"
GATED_MESSAGE = (
    "知识库中未找到相关内容：最相关片段的精排分数为 {score:.4f}，低于门控阈值 {threshold:.4f}。"
    "为避免在没有依据的情况下编造答案，这里不做回答。"
)


@dataclass
class Citation:
    """答案中的一条引用。"""

    index: int
    page_id: str  # 文章标识；界面据此定位图片所在的目录
    title: str
    section: str
    section_label: str
    source_url: str
    heading_path: str
    excerpt: str  # 被引用的父块原文（界面自行决定截断展示）


@dataclass
class AnswerResult:
    rejected: bool
    answer: str
    citations: list[Citation] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """一次检索编排的产物，供 ask 与评估共同消费。"""

    queries: list[str]  # 实际用于检索的 query（原 query 恒在首位）
    expanded: list[str]  # 额外展开的 query；空 = 单 query 模式（与现状一致）
    fused: list[ScoredHit]  # RRF 融合后的全量候选（按 chunk_id 去重）
    ranked: list[ScoredHit] = field(default_factory=list)  # 精排后的降序列表（多路模式下聚合分在 rerank_score）
    per_query: list[dict] = field(default_factory=list)  # 每个 query 的命中明细，供 debug 区分（F5）


def parse_citations(
    answer: str,
    entries: list[IndexEntry],
    labels: dict[str, str] | None = None,
) -> tuple[list[Citation], list[int]]:
    """把答案里的 [n] 映射回引用列表。

    返回（按首次出现顺序去重的引用, 越界的编号列表）。越界编号即悬空引用，
    由调用方记录用于验收核查。
    """
    ordered: list[int] = []
    seen: set[int] = set()
    dangling: list[int] = []
    for match in _CITATION.finditer(answer):
        index = int(match.group(1))
        if index < 1 or index > len(entries):
            if index not in dangling:
                dangling.append(index)
            continue
        if index not in seen:
            seen.add(index)
            ordered.append(index)

    citations = [
        Citation(
            index=index,
            page_id=entries[index - 1].page_id,
            title=entries[index - 1].title,
            section=entries[index - 1].section,
            section_label=section_label(entries[index - 1].section, labels),
            source_url=entries[index - 1].source_url,
            heading_path=entries[index - 1].heading_path,
            excerpt=entries[index - 1].parent_text,
        )
        for index in ordered
    ]
    return citations, dangling


def _dedupe_parents(entries: list[IndexEntry]) -> list[IndexEntry]:
    """多个子块可能落在同一父块，送入生成前按父块去重（保留排名最靠前的）。"""
    seen: set[str] = set()
    unique: list[IndexEntry] = []
    for entry in entries:
        if entry.parent_id in seen:
            continue
        seen.add(entry.parent_id)
        unique.append(entry)
    return unique


def rerank_aggregate(
    scoring_queries: list[str],
    candidate_entries: list[IndexEntry],
    cfg: Config,
    reranker,
) -> list[ScoredHit]:
    """用每个展开 query 对同一候选集打分，按子块取各 query 中的最高分聚合排序（F4）。

    max 聚合与门控语义对齐：任一聚焦角度认为相关内容足够相关即准入；广告块在所有
    聚焦 query 下都接近 0，真块在对应 query 下登顶，max 让真块胜出而广告自然降权
    （受控 probe 实测 0.90 vs 0.002~0.04）。原复合 query 不用于打分，只用于召回。
    未进任何 query 打分结果的候选不进入 ranked。
    """
    best_score: dict[str, float] = {}
    best_entry: dict[str, IndexEntry] = {}
    for query in scoring_queries:
        for hit in reranker.rerank(
            query, candidate_entries, cfg, top_n=len(candidate_entries)
        ):
            key = hit.entry.chunk_id
            score = hit.rerank_score if hit.rerank_score is not None else hit.score
            if score is None:
                continue
            if score > best_score.get(key, -1.0):
                best_score[key] = score
                best_entry[key] = hit.entry

    ordered = sorted(best_score, key=lambda key: (-best_score[key], key))
    return [
        ScoredHit(entry=best_entry[key], score=best_score[key], rerank_score=best_score[key])
        for key in ordered
    ]


def retrieve(
    question: str,
    cfg: Config,
    store: IndexStore,
    embedder=embedder_api,
    reranker=reranker_api,
    llm=llm_api,
    *,
    sections: list[str] | None = None,
) -> RetrievalResult:
    """多 Query 展开 → 多路召回 → RRF 融合 → 精排 的共享编排。

    单 query 模式（关闭/降级）与原链路逐字节一致；多路模式下任一展开 query 的检索
    失败只丢弃该路、不影响其它路（N4），但单 query 模式的失败照原样上抛（与现状一致）。
    """
    retrieval = cfg.retrieval
    queries, expanded = expand_queries(question, cfg, llm)
    multi = bool(expanded)

    per_query: list[dict] = []
    hit_lists: list[list[ScoredHit]] = []
    failed: list[str] = []
    for query in queries:
        try:
            vector = embedder.embed_query(query, cfg)
            vector_hits = store.search_vector(vector, retrieval.vec_top_k, sections)
            keyword_hits = store.search_bm25(query, retrieval.bm25_top_k, sections)
        except Exception as exc:  # noqa: BLE001 — 多路模式下单路失败不阻断
            if not multi:
                raise  # 单 query 模式与现状一致：失败上抛由调用方翻译
            logger.warning("展开 query 检索失败，跳过该路 %r：%s", query, exc)
            failed.append(query)
            continue
        per_query.append(
            {"query": query, "vector_hits": vector_hits, "keyword_hits": keyword_hits}
        )
        hit_lists.append(vector_hits)
        hit_lists.append(keyword_hits)

    fused = fuse_rrf_multi(hit_lists, retrieval.rrf_k)
    candidates = fused[: retrieval.rerank_top_n]
    candidate_entries = [hit.entry for hit in candidates]

    if multi:
        scoring = [q for q in expanded if q not in failed]
        if scoring:
            ranked = rerank_aggregate(scoring, candidate_entries, cfg, reranker)
        else:
            # 展开路全部失败：退化为按原 query 打分（与降级语义一致，F6）
            ranked = reranker.rerank(
                question, candidate_entries, cfg, top_n=len(candidate_entries)
            )
    else:
        ranked = reranker.rerank(
            question, candidate_entries, cfg, top_n=len(candidate_entries)
        )

    return RetrievalResult(
        queries=queries, expanded=expanded, fused=fused,
        ranked=ranked, per_query=per_query,
    )


def _channel_hits(result: RetrievalResult, key: str) -> list[ScoredHit]:
    """合并各 query 在某个通道的命中（按 chunk_id 去重，保留首次出现）。"""
    seen: set[str] = set()
    merged: list[ScoredHit] = []
    for item in result.per_query:
        for hit in item[key]:
            if hit.entry.chunk_id in seen:
                continue
            seen.add(hit.entry.chunk_id)
            merged.append(hit)
    return merged


def _debug_payload(
    result: RetrievalResult,
    selected: list[IndexEntry],
    threshold: float,
    full: bool,
    multi: bool,
) -> dict:
    summary = {
        "counts": {
            "vector": len(_channel_hits(result, "vector_hits")),
            "keyword": len(_channel_hits(result, "keyword_hits")),
            "fused": len(result.fused),
            "reranked": len(result.ranked),
            "selected": len(selected),
        },
        "gate_threshold": threshold,
        "top_rerank_score": result.ranked[0].rerank_score if result.ranked else None,
        "selected": [entry.chunk_id for entry in selected],
    }
    if not full:
        if multi:
            return {**summary, "queries": result.queries, "expanded": result.expanded}
        return summary

    payload = {
        **summary,
        "vector_hits": [
            (h.entry.chunk_id, round(h.score, 6)) for h in _channel_hits(result, "vector_hits")
        ],
        "keyword_hits": [
            (h.entry.chunk_id, round(h.score, 6)) for h in _channel_hits(result, "keyword_hits")
        ],
        "fused_hits": [(h.entry.chunk_id, round(h.score, 6)) for h in result.fused],
        "rerank_scores": [(h.entry.chunk_id, h.rerank_score) for h in result.ranked],
    }
    if multi:
        payload["queries"] = result.queries
        payload["expanded"] = result.expanded
        payload["per_query"] = [
            {
                "query": item["query"],
                "vector_hits": [
                    (h.entry.chunk_id, round(h.score, 6)) for h in item["vector_hits"]
                ],
                "keyword_hits": [
                    (h.entry.chunk_id, round(h.score, 6)) for h in item["keyword_hits"]
                ],
            }
            for item in result.per_query
        ]
    return payload


def ask(
    query: str,
    cfg: Config,
    store: IndexStore,
    embedder=embedder_api,
    reranker=reranker_api,
    llm=llm_api,
    sections: list[str] | None = None,
    debug: bool = False,
) -> AnswerResult:
    """回答一个问题：检索 → 门控 → 生成 → 引用。"""
    retrieval = cfg.retrieval
    question = query.strip()
    if not question:
        raise ValueError("问题不能为空")

    result = retrieve(question, cfg, store, embedder, reranker, llm, sections=sections)
    multi = bool(result.expanded)

    if not result.fused:
        return AnswerResult(
            rejected=True,
            answer=EMPTY_RETRIEVAL_MESSAGE,
            debug=_debug_payload(result, [], retrieval.gate_threshold, debug, multi),
        )

    payload = _debug_payload(result, [], retrieval.gate_threshold, debug, multi)
    top_score = (
        result.ranked[0].rerank_score
        if result.ranked and result.ranked[0].rerank_score is not None
        else 0.0
    )
    payload["top_rerank_score"] = top_score  # debug 里的「最高分」即门控分（多路=聚合 max 分）
    if top_score < retrieval.gate_threshold:
        return AnswerResult(
            rejected=True,
            answer=GATED_MESSAGE.format(score=top_score, threshold=retrieval.gate_threshold),
            debug=payload,
        )

    selected = _dedupe_parents([hit.entry for hit in result.ranked[: retrieval.final_n]])
    payload["selected"] = [entry.chunk_id for entry in selected]
    payload["counts"]["selected"] = len(selected)

    return _generate(question, selected, payload, cfg, llm, store.meta.get("section_labels"))


def _generate(
    question: str,
    entries: list[IndexEntry],
    payload: dict,
    cfg: Config,
    llm,
    labels: dict[str, str] | None = None,
) -> AnswerResult:
    """生成答案并解析引用，把检索细节并入 debug。"""
    answer = llm.chat(SYSTEM_PROMPT, build_user_prompt(question, entries, labels), cfg)
    citations, dangling = parse_citations(answer, entries, labels)
    payload["dangling_citations"] = dangling
    payload["cited"] = [citation.index for citation in citations]
    return AnswerResult(rejected=False, answer=answer, citations=citations, debug=payload)
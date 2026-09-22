"""离线评估：检索层 Hit@K / MRR + 生成层 LLM-as-a-Judge（对齐 rag/18_evaluation.md）。

- 检索层回答「该召回的有没有召回到、排得够不够靠前」
- 生成层回答「答案有没有幻觉、有没有跑题」
两层分开算，才能定位问题出在检索还是生成。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import embedder as embedder_api
from . import llm as llm_api
from . import reranker as reranker_api
from .config import Config
from .index_store import IndexEntry, IndexStore
from .pipeline import ask, retrieve

DEFAULT_SET_PATH = Path(__file__).resolve().parent.parent / "eval" / "set.jsonl"
HIT_K = 5
JUDGE_SAMPLE = 10
LOW_SCORE = 3  # 低于这个分数视为需要人工归因的坏例

JUDGE_SYSTEM = (
    "你是 RAG 系统的评估裁判。你要根据给定的参考资料，对答案的忠实度和相关性打分，"
    "只输出 JSON，不要输出多余解释。"
)

JUDGE_TEMPLATE = """请评估下面这个 RAG 答案。

【用户问题】
{question}

【参考资料】
{context}

【待评答案】
{answer}

请按两个维度各打 1-5 分（5 分最好）：
- faithfulness：答案里的每条论断是否都能在参考资料中找到依据。有参考资料之外的编造就给低分。
- answer_relevancy：答案是否切题、是否真正回答了用户的问题。跑题或答非所问给低分。

只输出 JSON，格式：{{"faithfulness": 4, "answer_relevancy": 5, "reason": "一句话说明"}}"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


class EvalError(Exception):
    """评估无法进行，消息面向使用者可直接阅读。"""


@dataclass
class EvalItem:
    item_id: str
    section: str
    page_id: str
    question: str
    hit: bool = False
    rank: int | None = None
    top_page_id: str = ""
    hit_before_rerank: bool = False


@dataclass
class JudgeScore:
    item_id: str
    question: str
    faithfulness: int
    answer_relevancy: int
    reason: str


@dataclass
class RejectCheck:
    """一条库外样本的拒答断言：门控确实把它挡下才算通过。"""

    item_id: str
    question: str
    rejected: bool
    top_score: float | None


@dataclass
class EvalReport:
    hit_at_k: float = 0.0
    mrr: float = 0.0
    hit_at_k_before_rerank: float = 0.0
    items: list[EvalItem] = field(default_factory=list)
    reject_checks: list[RejectCheck] = field(default_factory=list)
    judge_scores: list[JudgeScore] = field(default_factory=list)

    @property
    def judge_mean_faithfulness(self) -> float | None:
        if not self.judge_scores:
            return None
        return sum(s.faithfulness for s in self.judge_scores) / len(self.judge_scores)

    @property
    def judge_mean_relevancy(self) -> float | None:
        if not self.judge_scores:
            return None
        return sum(s.answer_relevancy for s in self.judge_scores) / len(self.judge_scores)

    @property
    def low_score_items(self) -> list[JudgeScore]:
        return [
            s for s in self.judge_scores if min(s.faithfulness, s.answer_relevancy) < LOW_SCORE
        ]


def load_eval_set(path: str | Path = DEFAULT_SET_PATH) -> list[dict]:
    """读取评测集，逐行校验必填字段。"""
    target = Path(path)
    if not target.exists():
        raise EvalError(f"未找到评测集 {target}")
    items = []
    for lineno, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise EvalError(f"{target} 第 {lineno} 行不是合法 JSON：{exc}") from exc
        for key in ("id", "section", "page_id", "question"):
            if not item.get(key):
                raise EvalError(f"{target} 第 {lineno} 行缺少字段 {key}")
        items.append(item)
    if not items:
        raise EvalError(f"{target} 中没有评测条目")
    return items


def entries_by_chunk_id(store: IndexStore) -> dict[str, IndexEntry]:
    return {entry.chunk_id: entry for entry in store.entries}


def evaluate_retrieval(
    items: list[dict],
    cfg: Config,
    store: IndexStore,
    embedder=embedder_api,
    reranker=reranker_api,
    llm=llm_api,
) -> list[EvalItem]:
    """逐题跑检索链路（多 Query 展开 → 多路召回 → RRF 融合 → 精排）。

    走全库口径（sections 恒 None，F7：跨栏目混淆要进入统计）；标了 expect_rejected 的
    库外样本不参与命中统计（F8，只由库外断言负责检查）。
    """
    results: list[EvalItem] = []
    for raw in items:
        question = raw["question"]
        target_page = raw["page_id"]
        if raw.get("expect_rejected"):
            continue

        result = retrieve(question, cfg, store, embedder, reranker, llm)
        ranked = result.ranked
        fused = result.fused

        item = EvalItem(
            item_id=raw["id"],
            section=raw["section"],
            page_id=target_page,
            question=question,
            top_page_id=ranked[0].entry.page_id if ranked else "",
            hit_before_rerank=any(h.entry.page_id == target_page for h in fused[:HIT_K]),
        )
        for rank, hit in enumerate(ranked[:HIT_K], start=1):
            if hit.entry.page_id == target_page:
                item.hit = True
                item.rank = rank
                break
        results.append(item)
    return results


def evaluate_rejections(
    items: list[dict],
    cfg: Config,
    store: IndexStore,
    embedder=embedder_api,
    reranker=reranker_api,
    llm=llm_api,
) -> list[RejectCheck]:
    """对每条库外样本跑完整问答，断言门控确实拒答（F8）。"""
    checks: list[RejectCheck] = []
    for raw in items:
        if not raw.get("expect_rejected"):
            continue
        result = ask(raw["question"], cfg, store, embedder, reranker, llm)
        checks.append(
            RejectCheck(
                item_id=raw["id"],
                question=raw["question"],
                rejected=result.rejected,
                top_score=result.debug.get("top_rerank_score"),
            )
        )
    return checks


def summarize(items: list[EvalItem]) -> tuple[float, float, float]:
    total = len(items)
    hit_at_k = sum(1 for i in items if i.hit) / total
    mrr = sum(1 / i.rank for i in items if i.hit and i.rank) / total
    before = sum(1 for i in items if i.hit_before_rerank) / total
    return hit_at_k, mrr, before


def parse_judge(raw: str) -> tuple[int, int, str] | None:
    """解析裁判输出，越界分数截断到 1-5。"""
    match = _JSON_OBJECT.search(raw)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        faithfulness = int(payload["faithfulness"])
        relevancy = int(payload["answer_relevancy"])
    except (KeyError, TypeError, ValueError):
        return None
    clamp = lambda value: max(1, min(5, value))  # noqa: E731
    return clamp(faithfulness), clamp(relevancy), str(payload.get("reason", "")).strip()


def judge_answers(
    items: list[dict],
    cfg: Config,
    store: IndexStore,
    sample: int = JUDGE_SAMPLE,
    llm=llm_api,
    **ask_kwargs,
) -> list[JudgeScore]:
    """抽样跑完整问答，把「问题 + 检索到的资料 + 答案」交给裁判模型打分。"""
    by_chunk = entries_by_chunk_id(store)
    scores: list[JudgeScore] = []
    for raw in items[:sample]:
        sections = [raw["section"]] if raw.get("section") else None
        result = ask(raw["question"], cfg, store, sections=sections, **ask_kwargs)
        if result.rejected:
            continue

        context = "\n\n".join(
            by_chunk[chunk_id].parent_text
            for chunk_id in result.debug.get("selected", [])
            if chunk_id in by_chunk
        )
        prompt = JUDGE_TEMPLATE.format(
            question=raw["question"], context=context, answer=result.answer
        )
        raw_score = llm.chat(JUDGE_SYSTEM, prompt, cfg, model=cfg.models.judge_model)
        parsed = parse_judge(raw_score)
        if parsed is None:
            continue
        faithfulness, relevancy, reason = parsed
        scores.append(
            JudgeScore(
                item_id=raw["id"],
                question=raw["question"],
                faithfulness=faithfulness,
                answer_relevancy=relevancy,
                reason=reason,
            )
        )
    return scores


def run_eval(
    cfg: Config,
    store: IndexStore,
    set_path: str | Path = DEFAULT_SET_PATH,
    judge_sample: int = JUDGE_SAMPLE,
    skip_judge: bool = False,
    embedder=embedder_api,
    reranker=reranker_api,
    llm=llm_api,
) -> EvalReport:
    """跑完整评估，返回报告；任一库外样本被放行即抛 EvalError（F8）。"""
    items = load_eval_set(set_path)
    report = EvalReport(items=evaluate_retrieval(items, cfg, store, embedder, reranker, llm))
    report.hit_at_k, report.mrr, report.hit_at_k_before_rerank = summarize(report.items)

    report.reject_checks = evaluate_rejections(items, cfg, store, embedder, reranker, llm)
    escaped = [c for c in report.reject_checks if not c.rejected]
    if escaped:
        detail = "；".join(
            f"{c.item_id}（最高分 {c.top_score if c.top_score is not None else 'N/A'}）"
            for c in escaped
        )
        raise EvalError(f"库外样本被放行，门控断言失败：{detail}")

    if not skip_judge:
        report.judge_scores = judge_answers(
            items, cfg, store, sample=judge_sample, llm=llm, embedder=embedder, reranker=reranker
        )
    return report


def format_report(report: EvalReport) -> str:
    """把报告渲染成可直接阅读的文本。"""
    lines = [
        "== 检索层 ==",
        f"题目数：{len(report.items)}",
        f"Hit@{HIT_K}：{report.hit_at_k:.3f}（目标 ≥ 0.800）",
        f"MRR（精排后）：{report.mrr:.3f}（目标 ≥ 0.500）",
        f"Hit@{HIT_K}（精排前/融合后）：{report.hit_at_k_before_rerank:.3f}",
        "",
        "逐题明细：",
    ]
    for item in report.items:
        flag = "命中" if item.hit else "未命中"
        rank = f"第 {item.rank} 名" if item.rank else f"top1 是 {item.top_page_id}"
        lines.append(f"  [{flag}] {item.item_id} {rank}  目标 {item.page_id}")

    lines += ["", "== 库外拒答断言 =="]
    if report.reject_checks:
        for check in report.reject_checks:
            verdict = "拒答" if check.rejected else "放行"
            score = f"{check.top_score:.4f}" if check.top_score is not None else "N/A"
            lines.append(f"  [{verdict}] {check.item_id} 最高分 {score}  {check.question}")
    else:
        lines.append("  无库外样本（跳过断言）")

    if report.judge_scores:
        lines += [
            "",
            "== 生成层（LLM 裁判，1-5 分）==",
            f"抽样数：{len(report.judge_scores)}",
            f"Faithfulness 均值：{report.judge_mean_faithfulness:.2f}（目标 ≥ 4.0）",
            f"Answer Relevancy 均值：{report.judge_mean_relevancy:.2f}（目标 ≥ 4.0）",
            "",
            "逐题打分：",
        ]
        for score in report.judge_scores:
            lines.append(
                f"  {score.item_id} 忠实 {score.faithfulness} 相关 {score.answer_relevancy}"
                f"  {score.reason}"
            )
        if report.low_score_items:
            lines += ["", f"需要人工归因的低分题（<{LOW_SCORE} 分）："]
            lines += [f"  {s.item_id} {s.question}" for s in report.low_score_items]
    else:
        lines += ["", "== 生成层 ==", "已跳过（--no-judge）"]

    return "\n".join(lines)

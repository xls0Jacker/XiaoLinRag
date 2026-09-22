"""多 Query 扩展：检索前把用户 query 拆成多个角度分别检索（docs 12 方法四 / 13 第三路 / 14 Multi-Query）。

原始 query 恒保留在检索列表首位（docs 14 硬约束：改写可能丢细节，原 query 最准）。
展开失败、返回空或全部重复一律回退为仅用原始 query，绝不阻断回答（F6 / N4）。
"""

from __future__ import annotations

import json
import logging

from .config import Config

logger = logging.getLogger(__name__)

EXPAND_SYSTEM = (
    "你负责把用户的问题拆解成多个不同角度的检索词，供知识库检索使用。"
    "只输出 JSON 数组，不要输出任何额外解释。"
)

EXPAND_TEMPLATE = """原始问题：{question}
请生成 {extra} 个不同角度的检索 query，每个是一个独立完整的中文问句或短语，
覆盖原问题可能涉及的各个子话题与考察维度，措辞尽量贴近技术文档表述，便于向量与关键词检索命中。
只输出 JSON 数组，如 ["检索词一", "检索词二", "检索词三"]。"""


def collapse_queries(queries: list[str]) -> list[str]:
    """清洗并去重：strip、丢空、按首次出现顺序去重。"""
    seen: set[str] = set()
    result: list[str] = []
    for raw in queries:
        item = raw.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def parse_expansion(raw: str) -> list[str]:
    """解析展开输出：JSON 数组 → 逐行 → 空列表，三级回退。"""
    text = raw.strip()
    if not text:
        return []

    start = text.find("[")
    if start >= 0:
        end = text.rfind("]")
        if end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except ValueError:
                pass
            else:
                if isinstance(payload, list):
                    return [item for item in payload if isinstance(item, str)]

    lines = [line.strip().strip("- '\"'，。") for line in text.splitlines() if line.strip()]
    # 逐行回退要 ≥2 行才认定为「换行列表」；单行散文无法与乱码区分，按失败处理
    return collapse_queries(lines) if len(lines) >= 2 else []


def _variant_count(cfg: Config) -> int:
    """需要 LLM 生成的展开数量（不含原始 query，总数为 num_queries）。"""
    return max(int(cfg.multi_query.num_queries) - 1, 1)


def expand_queries(question: str, cfg: Config, llm) -> tuple[list[str], list[str]]:
    """返回（全部检索 query, 额外展开 query）。

    关闭、数量为 1、LLM 抛错、解析为空或全部与原始问题重复时 → ([question], [])，
    此时调用方走单 query 链路、行为与现状一致（AC4 / AC6）。
    """
    question = question.strip()
    if not question:
        return [question], []
    mq = cfg.multi_query
    if not mq.enabled or mq.num_queries <= 1:
        return [question], []

    variants: list[str] = []
    try:
        raw = llm.chat(
            EXPAND_SYSTEM,
            EXPAND_TEMPLATE.format(question=question, extra=_variant_count(cfg)),
            cfg,
            temperature=mq.temperature,
        )
        variants = parse_expansion(raw)
    except Exception as exc:  # noqa: BLE001 — 展开失败一律降级，不阻断回答
        logger.warning("Query 展开失败，回退为仅用原始 query：%s", exc)
        return [question], []

    # 原 query 恒在首位；去重、丢弃与原始问题重复者、截断到 num_queries 条
    merged = collapse_queries([question] + variants)[: mq.num_queries]
    expanded = [q for q in merged if q != question]
    if not expanded:
        return [question], []
    return merged, expanded
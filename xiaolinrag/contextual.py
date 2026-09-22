"""Contextual Retrieval：入库前为每个子块补写背景说明（对齐 rag/5_semantic_cuts.md）。

背景说明与子块原文拼接后再向量化，缓解孤立子块「没头没尾」导致的召回偏差。
按文章成批生成（一篇文章一次调用），把 2000+ 次调用压到 100 次量级以满足构建耗时要求。
单篇失败只降级为空背景，不阻断整体构建（N5）。

传入复用源时，内容没变的文章直接沿用上一版索引里的说明，不再发起调用（见 reuse.py）。
提示词只吃文章正文与子块原文，所以「正文没变」是复用成立的充分条件。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from .chunking import ChildChunk
from .config import Config
from .corpus import Page
from .llm import chat

logger = logging.getLogger(__name__)

MAX_DOC_CHARS = 12000  # 超长文章截断，避免提示词过长
MAX_CHILDREN_PER_CALL = 30  # 单次调用最多处理多少个子块
MAX_ATTEMPTS = 2

SYSTEM_PROMPT = (
    "你是 RAG 索引构建助手。你的任务是为文档片段补写一句简短的背景说明，"
    "让后续的向量检索能理解这段内容在全文中的位置以及它在回答什么问题。"
)

USER_TEMPLATE = """下面是一篇技术文章的全文，以及从中切分出的若干片段。

<文章>
{doc}
</文章>

请为下面每个片段生成一句不超过 {limit} 字的背景说明：说明这段内容在全文中的位置，以及它在回答什么问题。
要求：独立成句、不要复述片段原文、不要输出「这段」「该片段」之外的指代。

<片段>
{listing}
</片段>

只输出 JSON 数组，形如 [{{"index": 0, "context": "..."}}, {{"index": 1, "context": "..."}}]，不要输出任何其他内容。"""

_JSON_ARRAY = re.compile(r"\[.*\]", re.S)
_LINE_PATTERN = re.compile(r"^\s*\[?(\d+)\]?\s*[:：]\s*(.+)$")


class _RateLimiter:
    """跨线程的请求间隔限速。"""

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last = time.monotonic()


def parse_contexts(raw: str) -> dict[int, str]:
    """从模型输出里解析 {片段下标: 背景说明}，先试 JSON，失败退回逐行解析。"""
    match = _JSON_ARRAY.search(raw)
    if match:
        try:
            payload = json.loads(match.group(0))
        except ValueError:
            payload = None
        if isinstance(payload, list):
            parsed = {}
            for item in payload:
                if not isinstance(item, dict) or "index" not in item:
                    continue
                try:
                    index = int(item["index"])
                except (TypeError, ValueError):
                    continue
                text = str(item.get("context") or "").strip()
                if text:
                    parsed[index] = text
            if parsed:
                return parsed

    parsed = {}
    for line in raw.splitlines():
        found = _LINE_PATTERN.match(line)
        if found:
            parsed[int(found.group(1))] = found.group(2).strip()
    return parsed


def _enrich_batch(
    page: Page, batch: list[ChildChunk], cfg: Config, chat_fn
) -> None:
    listing = "\n\n".join(
        f"[{i}] {chunk.child_text}" for i, chunk in enumerate(batch)
    )
    prompt = USER_TEMPLATE.format(
        doc=page.text[:MAX_DOC_CHARS],
        limit=cfg.chunking.context_max_chars,
        listing=listing,
    )

    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = chat_fn(SYSTEM_PROMPT, prompt, cfg)
        except Exception as exc:  # noqa: BLE001 — 降级为无背景，不阻断构建
            last_error = exc
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(1.5 * (2**attempt))
                continue
            break
        contexts = parse_contexts(raw)
        if contexts:
            for index, chunk in enumerate(batch):
                chunk.context_note = contexts.get(index, "")[: cfg.chunking.context_max_chars]
            return
        last_error = ValueError("模型输出无法解析出背景说明")

    logger.warning(
        "%s 的背景说明生成失败（%s），该批 %d 个子块将不带背景入库",
        page.page_id,
        last_error,
        len(batch),
    )


def _enrich_page(page: Page, chunks: list[ChildChunk], cfg: Config, chat_fn) -> None:
    for start in range(0, len(chunks), MAX_CHILDREN_PER_CALL):
        batch = chunks[start : start + MAX_CHILDREN_PER_CALL]
        _enrich_batch(page, batch, cfg, chat_fn)


def enrich(
    children: list[ChildChunk],
    pages: dict[str, Page],
    cfg: Config,
    chat_fn=None,
    reuse=None,
) -> None:
    """为全部子块原地填充 context_note。按文章并发，失败仅降级为空背景。

    传入 reuse（上一版索引的复用源）时，先按内容问它要该文的背景说明；命中就直接
    写回、不发调用，未命中才走分批调用。命中与否由 reuse 自己记账。
    """
    if not cfg.contextual.enabled or not children:
        return
    chat_fn = chat_fn or chat

    grouped: dict[str, list[ChildChunk]] = defaultdict(list)
    for chunk in children:
        grouped[chunk.page_id].append(chunk)

    limiter = _RateLimiter(cfg.contextual.interval_seconds)

    def work(item: tuple[str, list[ChildChunk]]) -> None:
        page_id, chunks = item
        page = pages.get(page_id)
        if page is None:
            logger.warning("%s 找不到对应文章，跳过背景生成", page_id)
            return
        if reuse is not None and _apply_reused(chunks, reuse, cfg):
            return
        limiter.wait()
        _enrich_page(page, chunks, cfg, chat_fn)

    with ThreadPoolExecutor(max_workers=cfg.contextual.concurrency) as pool:
        list(pool.map(work, sorted(grouped.items())))


def _apply_reused(chunks: list[ChildChunk], reuse, cfg: Config) -> bool:
    """把复用的背景说明写回子块；条数对不上就当作未命中（宁可重算也不用错位的说明）。"""
    key = reuse.context_key([chunk.child_text for chunk in chunks], cfg)
    notes = reuse.contexts(key)
    if notes is None or len(notes) != len(chunks):
        return False
    limit = cfg.chunking.context_max_chars
    for chunk, note in zip(chunks, notes):
        chunk.context_note = note[:limit]
    batches = (len(chunks) + MAX_CHILDREN_PER_CALL - 1) // MAX_CHILDREN_PER_CALL
    reuse.stats.context_calls_saved += batches
    return True

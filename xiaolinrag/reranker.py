"""Rerank 客户端（SiliconFlow /rerank）。

重排分数已由服务端归一化到 [0, 1]，可直接作为质量门控的阈值依据。
"""

from __future__ import annotations

import time

import httpx

from .config import Config
from .index_store import IndexEntry, ScoredHit

TIMEOUT_SECONDS = 120.0
MAX_ATTEMPTS = 3
RETRY_BASE_SECONDS = 1.5
MAX_DOCUMENTS = 100  # 单次请求的候选上限，超出由调用方先截断


class RerankError(Exception):
    """重排失败，消息面向使用者可直接阅读。"""


def _brief(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:200]
    if isinstance(payload, dict):
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:200]
    return str(payload)[:200]


def rerank(
    query: str,
    entries: list[IndexEntry],
    cfg: Config,
    top_n: int | None = None,
) -> list[ScoredHit]:
    """对候选子块精排，按相关度降序返回。"""
    if not entries:
        return []
    if not query.strip():
        raise RerankError("查询内容为空，无法重排")

    documents = [entry.child_text for entry in entries]
    payload = {
        "model": cfg.models.rerank_model,
        "query": query.strip(),
        "documents": documents,
        "top_n": min(top_n or len(documents), len(documents)),
        "return_documents": False,
    }
    url = cfg.models.rerank_base_url.rstrip("/") + "/rerank"
    headers = {"Authorization": f"Bearer {cfg.models.embedding_api_key}"}

    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < MAX_ATTEMPTS - 1:
                time.sleep(RETRY_BASE_SECONDS * (2**attempt))
                continue
            if response.status_code == 401:
                raise RerankError(
                    f"重排鉴权失败（{cfg.models.rerank_base_url}），"
                    "请检查 .env 里的 SILICONFLOW_API_KEY"
                )
            if response.status_code >= 400:
                raise RerankError(
                    f"重排服务返回错误（HTTP {response.status_code}）：{_brief(response)}"
                )
            data = response.json()
            results = data.get("results")
            if not results:
                raise RerankError("重排服务未返回任何结果")
            return [
                ScoredHit(
                    entry=entries[int(item["index"])],
                    score=float(item["relevance_score"]),
                    rerank_score=float(item["relevance_score"]),
                )
                for item in results
            ]
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(RETRY_BASE_SECONDS * (2**attempt))
                continue
            raise RerankError(f"无法连接重排服务 {cfg.models.rerank_base_url}，请检查网络") from exc

    raise RerankError(f"重排多次重试后仍失败：{last_error}")

"""Embedding 客户端（SiliconFlow）。

Qwen3-Embedding 系列按官方惯例区分 query 与 document：query 加指令前缀，
document 不加。返回值统一做 L2 归一化，供内积索引当余弦相似度使用。
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from .config import Config

TIMEOUT_SECONDS = 120.0
BATCH_SIZE = 32
MAX_ATTEMPTS = 3
RETRY_BASE_SECONDS = 1.5

# Qwen3-Embedding 的检索任务指令前缀，仅用于查询侧
QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class EmbeddingError(Exception):
    """向量化失败，消息面向使用者可直接阅读。"""


@lru_cache(maxsize=8)
def _client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key, timeout=TIMEOUT_SECONDS)


def _brief(exc: APIStatusError) -> str:
    message = getattr(exc, "message", None) or str(exc)
    return message.strip().splitlines()[0][:200]


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


def _translate(exc: Exception, cfg: Config) -> EmbeddingError:
    if isinstance(exc, AuthenticationError):
        return EmbeddingError(
            f"向量化鉴权失败（{cfg.models.embedding_base_url}），请检查 .env 里的 SILICONFLOW_API_KEY"
        )
    if isinstance(exc, RateLimitError):
        return EmbeddingError("向量化触发限流，请稍后重试")
    if isinstance(exc, APITimeoutError):
        return EmbeddingError(f"向量化请求超时（{TIMEOUT_SECONDS:.0f} 秒）")
    if isinstance(exc, APIConnectionError):
        return EmbeddingError(f"无法连接向量化服务 {cfg.models.embedding_base_url}，请检查网络")
    if isinstance(exc, APIStatusError):
        return EmbeddingError(f"向量化服务返回错误（HTTP {exc.status_code}）：{_brief(exc)}")
    return EmbeddingError(f"向量化失败：{exc}")


def _embed(texts: list[str], cfg: Config, on_progress=None) -> np.ndarray:
    """批量向量化并 L2 归一化，按批分片、失败退避重试。"""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    client = _client(cfg.models.embedding_base_url, cfg.models.embedding_api_key)
    vectors: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = [t if t.strip() else " " for t in texts[start : start + BATCH_SIZE]]
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = client.embeddings.create(
                    model=cfg.models.embedding_model, input=batch
                )
                break
            except Exception as exc:  # noqa: BLE001 — 统一翻译为 EmbeddingError
                if _is_retryable(exc) and attempt < MAX_ATTEMPTS - 1:
                    time.sleep(RETRY_BASE_SECONDS * (2**attempt))
                    continue
                raise _translate(exc, cfg) from exc
        vectors.extend(item.embedding for item in response.data)
        if on_progress is not None:
            on_progress(min(start + BATCH_SIZE, len(texts)), len(texts))

    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def embed_documents(texts: list[str], cfg: Config, on_progress=None) -> np.ndarray:
    """向量化用于入库的文本（子块），不加查询指令前缀。

    on_progress(done, total) 用于构建时展示进度。
    """
    return _embed(texts, cfg, on_progress=on_progress)


def embed_query(query: str, cfg: Config) -> np.ndarray:
    """向量化用户查询，加 Qwen3 检索指令前缀。"""
    if not query.strip():
        raise EmbeddingError("查询内容为空，无法向量化")
    prefixed = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query.strip()}"
    return _embed([prefixed], cfg)[0]

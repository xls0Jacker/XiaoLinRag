"""对话模型客户端：OpenAI 兼容网关（当前为 OpenCode Go）。

错误统一翻译成面向使用者的中文提示（N5），且任何提示都不包含密钥原文（N2）。
"""

from __future__ import annotations

from functools import lru_cache

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from .config import CHAT_KEY_ENV, Config

TIMEOUT_SECONDS = 120.0


class LLMError(Exception):
    """对话模型调用失败，消息面向使用者可直接阅读。"""


@lru_cache(maxsize=8)
def _client(base_url: str, api_key: str, extra_headers: tuple[tuple[str, str], ...] = ()) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=TIMEOUT_SECONDS,
        default_headers=dict(extra_headers) or None,
    )


def _brief(exc: APIStatusError) -> str:
    """从服务端错误里取一句短描述，不回显请求头与密钥。"""
    message = getattr(exc, "message", None) or str(exc)
    return message.strip().splitlines()[0][:200]


def chat(
    system: str,
    user: str,
    cfg: Config,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """一轮对话补全，返回助手回复正文。"""
    client = _client(
        cfg.models.chat_base_url,
        cfg.models.chat_api_key,
        tuple(sorted(cfg.models.chat_extra_headers.items())),
    )
    kwargs: dict = {
        "model": model or cfg.models.chat_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": cfg.models.chat_temperature if temperature is None else temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        response = client.chat.completions.create(**kwargs)
    except AuthenticationError as exc:
        raise LLMError(
            f"对话模型鉴权失败（{cfg.models.chat_base_url}），请检查 .env 里的 {CHAT_KEY_ENV}"
        ) from exc
    except RateLimitError as exc:
        raise LLMError("对话模型触发限流，请稍后重试") from exc
    except APITimeoutError as exc:
        raise LLMError(f"对话模型请求超时（{TIMEOUT_SECONDS:.0f} 秒）") from exc
    except APIConnectionError as exc:
        raise LLMError(f"无法连接对话模型服务 {cfg.models.chat_base_url}，请检查网络") from exc
    except APIStatusError as exc:
        raise LLMError(f"对话模型返回错误（HTTP {exc.status_code}）：{_brief(exc)}") from exc

    content = response.choices[0].message.content if response.choices else None
    if not content or not content.strip():
        raise LLMError("对话模型返回了空内容")
    return content.strip()

from pathlib import Path

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
from xiaolinrag.multi_query import collapse_queries, expand_queries, parse_expansion


class FakeLLM:
    def __init__(self, replies=None, error=None):
        self.replies = list(replies or [])
        self.error = error
        self.calls = []

    def chat(self, system, user, cfg, **kwargs):
        self.calls.append((system, user, kwargs))
        if self.error is not None:
            raise self.error
        if self.replies:
            return self.replies.pop(0)
        return ""


def make_cfg(enabled=True, num_queries=4, temperature=0.2) -> Config:
    return Config(
        kb_dir=Path("/tmp/kb"),
        index_dir=Path("/tmp/index"),
        models=ModelConfig("u", "m", "u", "m", "u", "m", "m", 0.3, "sk", "sk"),
        chunking=ChunkingConfig(300, 500, 900, 1200, 100),
        contextual=ContextualConfig(True, 4, 0.0),
        retrieval=RetrievalConfig(20, 20, 60, 40, 5, 0.5),
        webui=WebUIConfig("127.0.0.1", 7860),
        multi_query=MultiQueryConfig(enabled, num_queries, temperature),
    )


Q = "三次握手和四次挥手分别是什么"


def test_disabled_does_not_call_llm():
    llm = FakeLLM(error=AssertionError("不应调用 LLM"))
    queries, expanded = expand_queries(Q, make_cfg(enabled=False), llm)

    assert queries == [Q]
    assert expanded == []
    assert llm.calls == []


def test_num_queries_one_does_not_call_llm():
    llm = FakeLLM(error=AssertionError("不应调用 LLM"))
    queries, expanded = expand_queries(Q, make_cfg(num_queries=1), llm)

    assert queries == [Q]
    assert expanded == []


def test_llm_error_degrades_to_original():
    llm = FakeLLM(error=LLMError("服务不可用"))
    queries, expanded = expand_queries(Q, make_cfg(), llm)

    assert queries == [Q]
    assert expanded == []
    assert len(llm.calls) == 1


def test_original_query_always_first():
    llm = FakeLLM(replies=['["握手过程", "挥手过程", "TIME_WAIT 排查"]'])
    queries, expanded = expand_queries(Q, make_cfg(), llm)

    assert queries[0] == Q
    assert expanded == ["握手过程", "挥手过程", "TIME_WAIT 排查"]


def test_duplicate_variants_deduped():
    llm = FakeLLM(replies=["[\"握手过程\", \"握手过程\", \"挥手过程\"]"])
    queries, expanded = expand_queries(Q, make_cfg(num_queries=4), llm)

    assert queries == [Q, "握手过程", "挥手过程"]
    assert expanded == ["握手过程", "挥手过程"]


def test_count_capped_at_num_queries():
    llm = FakeLLM(replies=['["a", "b", "c", "d"]'])
    queries, expanded = expand_queries(Q, make_cfg(num_queries=3), llm)

    assert len(queries) == 3
    assert queries == [Q, "a", "b"]
    assert expanded == ["a", "b"]


def test_all_variants_same_as_question_falls_back():
    llm = FakeLLM(replies=["[\"三次握手和四次挥手分别是什么\"]"])
    queries, expanded = expand_queries(Q, make_cfg(num_queries=2), llm)

    assert queries == [Q]
    assert expanded == []


def test_garbage_reply_falls_back():
    llm = FakeLLM(replies=["这不是一个可解析的展开结果"])
    queries, expanded = expand_queries(Q, make_cfg(), llm)

    assert queries == [Q]
    assert expanded == []


def test_reply_honors_low_temperature():
    llm = FakeLLM(replies=['["握手过程"]'])
    expand_queries(Q, make_cfg(temperature=0.0), llm)

    assert llm.calls[0][2]["temperature"] == 0.0


# ---------- parse / collapse 单测 ----------


def test_parse_json_array_with_surrounding_prose():
    out = parse_expansion('解释一下。["握手过程", "挥手过程"] 完毕')
    assert out == ["握手过程", "挥手过程"]


def test_parse_line_per_query():
    out = parse_expansion("握手过程\n挥手过程\n- TIME_WAIT 排查")
    assert out == ["握手过程", "挥手过程", "TIME_WAIT 排查"]


def test_parse_empty_and_garbage():
    assert parse_expansion("") == []
    assert parse_expansion("随便一句话，没有数组也没有分行") == []


def test_collapse_queries_strip_and_dedupe():
    assert collapse_queries([" a ", "a", "", "b", "b", "  c  "]) == ["a", "b", "c"]
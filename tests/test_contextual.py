from xiaolinrag.chunking import ChildChunk
from xiaolinrag.config import (
    ChunkingConfig,
    Config,
    ContextualConfig,
    ModelConfig,
    RetrievalConfig,
    WebUIConfig,
)
from xiaolinrag.contextual import enrich, parse_contexts
from xiaolinrag.corpus import Page
from pathlib import Path
from types import SimpleNamespace


def make_cfg(enabled: bool = True, context_max_chars: int = 100) -> Config:
    return Config(
        kb_dir=Path("/tmp/kb"),
        index_dir=Path("/tmp/index"),
        models=ModelConfig("u", "m", "u", "m", "u", "m", "m", 0.3, "k1", "k2"),
        chunking=ChunkingConfig(300, 500, 900, 1200, context_max_chars),
        contextual=ContextualConfig(enabled, 1, 0.0),
        retrieval=RetrievalConfig(20, 20, 60, 40, 5, 0.3),
        webui=WebUIConfig("127.0.0.1", 7860),
    )


def make_page(page_id: str = "xiaolinnote/rag/demo") -> Page:
    return Page(
        page_id=page_id,
        site="xiaolinnote",
        section="xiaolinnote/rag",
        title="示例文章",
        source_url="https://example.com",
        text="# 示例文章\n\n正文内容。\n",
    )


def make_children(count: int, page_id: str = "xiaolinnote/rag/demo") -> list[ChildChunk]:
    return [
        ChildChunk(
            chunk_id=f"{page_id}#c{i}",
            page_id=page_id,
            heading_path="示例文章",
            child_text=f"第 {i} 个子块原文",
            parent_id=f"{page_id}#p0",
            seq=i,
        )
        for i in range(count)
    ]


def test_parse_json_array():
    raw = '[{"index": 0, "context": "背景A"}, {"index": 1, "context": "背景B"}]'

    assert parse_contexts(raw) == {0: "背景A", 1: "背景B"}


def test_parse_json_with_surrounding_prose():
    raw = '好的，结果如下：\n[{"index": 2, "context": "背景C"}]\n希望有帮助。'

    assert parse_contexts(raw) == {2: "背景C"}


def test_parse_skips_empty_and_malformed_items():
    raw = '[{"index": 0, "context": "有"}, {"context": "缺index"}, {"index": 2, "context": "  "}]'

    assert parse_contexts(raw) == {0: "有"}


def test_parse_falls_back_to_lines():
    raw = "0: 这是背景一\n[1]：这是背景二"

    assert parse_contexts(raw) == {0: "这是背景一", 1: "这是背景二"}


def test_parse_returns_empty_on_garbage():
    assert parse_contexts("模型不太确定该说什么") == {}


def test_parse_empty_input():
    assert parse_contexts("") == {}


def test_enrich_fills_context_notes():
    children = make_children(2)
    cfg = make_cfg()

    def fake_chat(system, user, cfg):
        assert "第 0 个子块原文" in user and "正文内容" in user
        return '[{"index": 0, "context": "背景0"}, {"index": 1, "context": "背景1"}]'

    enrich(children, {"xiaolinnote/rag/demo": make_page()}, cfg, chat_fn=fake_chat)

    assert [c.context_note for c in children] == ["背景0", "背景1"]


def test_enrich_truncates_to_limit():
    children = make_children(1)
    cfg = make_cfg(context_max_chars=5)

    enrich(
        children,
        {"xiaolinnote/rag/demo": make_page()},
        cfg,
        chat_fn=lambda s, u, c: '[{"index": 0, "context": "一二三四五六七八九十"}]',
    )

    assert children[0].context_note == "一二三四五"


def test_enrich_degrades_when_model_fails():
    children = make_children(2)
    cfg = make_cfg()

    def boom(system, user, cfg):
        raise RuntimeError("模型不可用")

    enrich(children, {"xiaolinnote/rag/demo": make_page()}, cfg, chat_fn=boom)

    assert [c.context_note for c in children] == ["", ""]  # 降级为空背景，不抛错


def test_enrich_degrades_when_output_unparseable():
    children = make_children(1)
    cfg = make_cfg()

    enrich(children, {"xiaolinnote/rag/demo": make_page()}, cfg, chat_fn=lambda s, u, c: "我答不上来")

    assert children[0].context_note == ""


def test_enrich_skips_when_disabled():
    children = make_children(1)
    cfg = make_cfg(enabled=False)
    calls = []

    enrich(
        children,
        {"xiaolinnote/rag/demo": make_page()},
        cfg,
        chat_fn=lambda s, u, c: calls.append(1) or "[]",
    )

    assert calls == []
    assert children[0].context_note == ""


def test_enrich_skips_unknown_page():
    children = make_children(1)

    enrich(children, {}, make_cfg(), chat_fn=lambda s, u, c: "[]")

    assert children[0].context_note == ""


def test_enrich_splits_large_page_into_batches():
    children = make_children(31)  # MAX_CHILDREN_PER_CALL = 30
    cfg = make_cfg()
    batches = []

    def fake_chat(system, user, cfg):
        size = user.count("个子块原文")
        batches.append(size)
        items = ", ".join(f'{{"index": {i}, "context": "背景{i}"}}' for i in range(size))
        return f"[{items}]"

    enrich(children, {"xiaolinnote/rag/demo": make_page()}, cfg, chat_fn=fake_chat)

    assert batches == [30, 1], "超过 30 个子块应拆成两次调用"
    assert all(c.context_note for c in children)


class FakeReuse:
    """只实现 enrich 用到的那几个接口。"""

    def __init__(self, notes):
        self.notes = notes
        self.lookups = 0
        self.stats = SimpleNamespace(context_calls_saved=0)

    def context_key(self, child_texts, cfg):
        return f"key:{len(child_texts)}"

    def contexts(self, key):
        self.lookups += 1
        return self.notes


def test_enrich_reuses_notes_without_calling_model():
    children = make_children(2)
    calls = []
    reuse = FakeReuse(["复用来的背景一", "复用来的背景二"])

    enrich(
        children,
        {"xiaolinnote/rag/demo": make_page()},
        make_cfg(),
        chat_fn=lambda s, u, c: calls.append(1) or "[]",
        reuse=reuse,
    )

    assert calls == [], "命中复用就不该再调用对话模型"
    assert [c.context_note for c in children] == ["复用来的背景一", "复用来的背景二"]
    assert reuse.lookups == 1
    assert reuse.stats.context_calls_saved == 1


def test_enrich_truncates_reused_notes():
    children = make_children(1)
    reuse = FakeReuse(["长" * 500])

    enrich(children, {"xiaolinnote/rag/demo": make_page()}, make_cfg(context_max_chars=10), reuse=reuse)

    assert len(children[0].context_note) == 10


def test_enrich_regenerates_when_reuse_count_mismatches():
    """复用回来的说明条数与子块数对不上时宁可重算，不用错位的说明。"""
    children = make_children(2)
    reuse = FakeReuse(["只有一条"])

    def fake_chat(system, user, cfg):
        size = user.count("个子块原文")
        items = ", ".join(f'{{"index": {i}, "context": "重新生成{i}"}}' for i in range(size))
        return f"[{items}]"

    enrich(
        children,
        {"xiaolinnote/rag/demo": make_page()},
        make_cfg(),
        chat_fn=fake_chat,
        reuse=reuse,
    )

    assert [c.context_note for c in children] == ["重新生成0", "重新生成1"]


def test_enrich_without_reuse_behaves_as_before():
    children = make_children(2)

    def fake_chat(system, user, cfg):
        size = user.count("个子块原文")
        items = ", ".join(f'{{"index": {i}, "context": "背景{i}"}}' for i in range(size))
        return f"[{items}]"

    enrich(children, {"xiaolinnote/rag/demo": make_page()}, make_cfg(), chat_fn=fake_chat, reuse=None)

    assert [c.context_note for c in children] == ["背景0", "背景1"]

from pathlib import Path

from xiaolinrag.chunking import _blocks, split_page
from xiaolinrag.config import ChunkingConfig
from xiaolinrag.corpus import Page, load_pages

FIXTURE_KB = Path(__file__).parent / "fixtures" / "kb"

CFG = ChunkingConfig(
    child_target=200,
    child_max=300,
    parent_target=400,
    parent_max=600,
    context_max_chars=100,
)


def fixture_page(page_id: str = "xiaolinnote/rag/sample_chunking") -> Page:
    return {p.page_id: p for p in load_pages(FIXTURE_KB)}[page_id]


def find_child(children, keyword: str):
    hits = [c for c in children if keyword in c.child_text]
    assert hits, f"没有子块包含 {keyword!r}"
    return hits[0]


def test_heading_path_follows_structure():
    children, _ = split_page(fixture_page(), CFG)

    child = find_child(children, "表格要整块保留")

    assert child.heading_path == "示例：文档切割策略 > 语义边界切割 > 特殊内容处理"
    assert child.child_text.startswith("特殊内容处理")


def test_code_block_kept_whole():
    children, _ = split_page(fixture_page(), CFG)

    hits = [c for c in children if "def split(text, size)" in c.child_text]

    assert len(hits) == 1, "代码块被切散到多个子块"
    assert "```python" in hits[0].child_text
    assert "for i in range(0, len(text), size)]" in hits[0].child_text


def test_table_kept_whole():
    children, _ = split_page(fixture_page(), CFG)

    child = find_child(children, "| 策略 | 适用文档 | 优点 |")

    assert "| 固定大小 | 纯文本 | 实现简单 |" in child.child_text
    assert "| 语义边界 | 段落分明的文章 | 语义完整 |" in child.child_text


def test_child_belongs_to_parent():
    children, parents = split_page(fixture_page(), CFG)
    by_id = {p.parent_id: p for p in parents}

    for child in children:
        assert child.parent_id in by_id, f"{child.chunk_id} 未关联父块"
        assert child.child_text in by_id[child.parent_id].parent_text
        assert child.chunk_id in by_id[child.parent_id].child_ids


def test_parent_uses_every_child_once():
    children, parents = split_page(fixture_page(), CFG)

    linked = [cid for p in parents for cid in p.child_ids]

    assert sorted(linked) == sorted(c.chunk_id for c in children)


def test_child_size_limit():
    children, _ = split_page(fixture_page(), CFG)

    assert children
    assert max(len(c.child_text) for c in children) <= CFG.child_max


def test_parent_size_limit():
    _, parents = split_page(fixture_page(), CFG)

    for parent in parents:
        if len(parent.child_ids) > 1:
            assert len(parent.parent_text) <= CFG.parent_max


def test_oversized_code_block_is_atomic():
    code = "\n".join(f"    line_{i} = {i} * 2" for i in range(60))
    page = Page(
        page_id="rag/big_code",
        site="xiaolinnote",
        section="xiaolinnote/rag",
        title="大代码块",
        source_url="https://example.com",
        text=f"# 大代码块\n\n说明文字。\n\n```python\n{code}\n```\n\n结尾文字。\n",
    )

    children, parents = split_page(page, CFG)

    holder = [c for c in children if "line_59" in c.child_text]
    assert len(holder) == 1
    assert "line_0 = 0 * 2" in holder[0].child_text, "长代码块被截断"
    assert len(holder[0].child_text) > CFG.child_max, "长代码块应整块保留、允许超限"
    assert any(holder[0].child_text in p.parent_text for p in parents)


def test_oversized_paragraph_falls_back_to_sentences():
    sentence = "这是一个用来测试句子边界回退的完整句子。"
    page = Page(
        page_id="rag/long_para",
        site="xiaolinnote",
        section="xiaolinnote/rag",
        title="超长段落",
        source_url="https://example.com",
        text="# 超长段落\n\n" + sentence * 30 + "\n",
    )

    children, _ = split_page(page, CFG)

    assert len(children) > 1, "超长段落未按句子边界回退切分"
    for child in children:
        assert len(child.child_text) <= CFG.child_max
        assert child.child_text.rstrip().endswith("。")


def test_heading_only_page_yields_nothing():
    page = Page(
        page_id="rag/empty",
        site="xiaolinnote",
        section="xiaolinnote/rag",
        title="空页",
        source_url="https://example.com",
        text="# 只有标题\n",
    )

    assert split_page(page, CFG) == ([], [])


def test_blocks_keeps_list_markers():
    page = fixture_page()

    kinds = [b.kind for b in _blocks(page.text)]

    assert "heading" in kinds
    assert "code" in kinds
    assert "list" in kinds
    assert "table" in kinds
    list_block = next(b for b in _blocks(page.text) if b.kind == "list")
    assert list_block.text.startswith("- "), "列表符号丢失"

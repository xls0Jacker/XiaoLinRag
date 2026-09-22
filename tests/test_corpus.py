import json
import shutil
from pathlib import Path

import pytest

from xiaolinrag.corpus import CorpusError, load_pages, load_sites

FIXTURE_KB = Path(__file__).parent / "fixtures" / "kb"


@pytest.fixture
def kb(tmp_path):
    """把 fixture 知识库复制到临时目录，便于制造异常场景。"""
    dst = tmp_path / "kb"
    shutil.copytree(FIXTURE_KB, dst)
    return dst


def test_load_all_pages(kb):
    pages = load_pages(kb)

    assert [p.page_id for p in pages] == [
        "xiaolincoding/network/1_base/sample_tcp",
        "xiaolinnote/llm/sample_llm",
        "xiaolinnote/rag/sample_chunking",
        "xiaolinnote/rag/sample_short",
    ]


def test_page_fields(kb):
    pages = {p.page_id: p for p in load_pages(kb)}
    page = pages["xiaolinnote/rag/sample_chunking"]

    assert page.site == "xiaolinnote"
    assert page.section == "xiaolinnote/rag"  # 复合键：站点/栏目
    assert page.title == "示例：文档切割策略"
    assert page.source_url == "https://xiaolinnote.com/ai/rag/sample_chunking.html"
    assert page.text.startswith("# 示例：文档切割策略")
    assert "---" not in page.text.splitlines()[0]  # frontmatter 已剥离
    assert "固定大小切割" in page.text


def test_nested_directory_page_is_discovered(kb):
    """栏目下还有章节子目录时，深层的文章同样要被发现。"""
    pages = {p.page_id: p for p in load_pages(kb)}
    page = pages["xiaolincoding/network/1_base/sample_tcp"]

    assert page.site == "xiaolincoding"
    assert page.section == "xiaolincoding/network"
    assert "三次握手" in page.text


def test_load_sites_reports_labels_and_order(kb):
    sites = load_sites(kb)

    assert [s.key for s in sites] == ["xiaolinnote", "xiaolincoding"]  # 按索引声明顺序
    assert [s.label for s in sites] == ["小林面试笔记", "小林coding"]
    assert [s.key for s in sites[0].sections] == ["xiaolinnote/rag", "xiaolinnote/llm"]
    assert [s.label for s in sites[0].sections] == ["RAG", "LLM"]
    assert sites[1].sections[0].label == "图解网络"


def test_missing_manifest(tmp_path):
    with pytest.raises(CorpusError, match="manifest.json"):
        load_pages(tmp_path)


def test_manifest_without_sites_is_reported(kb):
    """旧版索引只有顶层 sections，没有 sites——要给出可读提示而不是 KeyError。"""
    path = kb / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("sites")
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(CorpusError, match="sites"):
        load_pages(kb)


def test_missing_file_reported(kb):
    (kb / "xiaolinnote" / "rag" / "sample_short.md").unlink()

    with pytest.raises(CorpusError, match="索引里声明但磁盘上没有"):
        load_pages(kb)


def test_extra_file_reported(kb):
    (kb / "xiaolinnote" / "rag" / "unexpected.md").write_text(
        "---\ntitle: x\nsource_url: y\n---\n\n正文\n", encoding="utf-8"
    )

    with pytest.raises(CorpusError, match="磁盘上多出但索引未收录"):
        load_pages(kb)


def test_markdown_under_assets_is_not_corpus(kb):
    """assets/ 下是图片资源目录，即使混进 .md 也不算文章。"""
    assets = kb / "xiaolinnote" / "rag" / "assets" / "sample_chunking"
    assets.mkdir(parents=True)
    (assets / "readme.md").write_text("这不是文章\n", encoding="utf-8")

    assert len(load_pages(kb)) == 4


def test_missing_section_dir(kb):
    shutil.rmtree(kb / "xiaolinnote" / "llm")

    with pytest.raises(CorpusError, match="栏目目录不存在"):
        load_pages(kb)


def test_missing_frontmatter(kb):
    (kb / "xiaolinnote" / "rag" / "sample_short.md").write_text(
        "# 没有 frontmatter\n", encoding="utf-8"
    )

    with pytest.raises(CorpusError, match="缺少 frontmatter"):
        load_pages(kb)


def test_frontmatter_missing_title(kb):
    (kb / "xiaolinnote" / "rag" / "sample_short.md").write_text(
        "---\nsource_url: https://example.com\n---\n\n正文\n", encoding="utf-8"
    )

    with pytest.raises(CorpusError, match="title"):
        load_pages(kb)

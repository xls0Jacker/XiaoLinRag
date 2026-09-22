"""前端纯函数的回归测试：用 node 直接跑 app.js 导出的函数。

转义是安全相关行为（spec N3/AC12），靠肉眼在浏览器里看一次防不住回归，
所以让 pytest 驱动 node 跑对抗性输入。node 不在时整份跳过，不阻塞其余测试。
"""

import json
import re
import shutil
import subprocess

import pytest

from xiaolinrag import api

APP_JS = api.STATIC_DIR / "app.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="未安装 node，跳过前端纯函数测试"
)

PROBE = """
import { readFileSync } from "node:fs";
const mod = await import(process.argv[2]);
const payload = JSON.parse(readFileSync(0, "utf8"));

// 与前端调用处同构的两个解析器替身
const resolveImage = (src) => (src.startsWith("assets/") ? `/kb/rag/${src}` : null);
const resolveLink = (href) => {
  try {
    const url = new URL(href, "https://xiaolinnote.com/ai/rag/x.html");
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
};

process.stdout.write(JSON.stringify({
  markdown: payload.markdown.map((text) => mod.renderMarkdown(text)),
  escape: payload.escape.map((text) => mod.escapeHtml(text)),
  image: (payload.image || []).map((text) => mod.renderMarkdown(text, { resolveImage })),
  link: (payload.link || []).map((text) => mod.renderMarkdown(text, { resolveLink })),
  citation: (payload.citation || []).map(([pageId, src]) => mod.citationImageUrl(pageId, src)),
}));
"""

# 渲染器允许产出的全部标签。除此之外输出里不该再出现裸标记。
ALLOWED_TAG = re.compile(
    "|".join(
        [
            r'<a class="cite" href="#cite-\d+">',
            r'<a href="https?://[^"]*" target="_blank" rel="noopener noreferrer">',
            r'<img src="[^"]*" alt="[^"]*" loading="lazy" decoding="async">',
            r'<div class="table-wrap">',
            r'<t[hd](?: style="text-align:(?:left|right|center)")?>',
            r"</?(?:p|strong|em|del|code|pre|ul|ol|li|a|blockquote|table|thead|tbody|tr|th|td|div|h[2-6])>",
        ]
    )
)


def run_frontend(
    tmp_path, markdown_inputs=(), escape_inputs=(), image_inputs=(), link_inputs=(),
    citation_inputs=(),
):
    """把输入交给 node 里的 app.js，拿回纯函数的输出。"""
    probe = tmp_path / "probe.mjs"
    probe.write_text(PROBE, encoding="utf-8")
    completed = subprocess.run(
        ["node", str(probe), APP_JS.as_uri()],
        input=json.dumps(
            {
                "markdown": list(markdown_inputs),
                "escape": list(escape_inputs),
                "image": list(image_inputs),
                "link": list(link_inputs),
                "citation": [list(pair) for pair in citation_inputs],
            }
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def render(tmp_path, *texts):
    return run_frontend(tmp_path, markdown_inputs=texts)["markdown"]


def render_with_image_resolver(tmp_path, *texts):
    return run_frontend(tmp_path, image_inputs=texts)["image"]


def render_with_link_resolver(tmp_path, *texts):
    return run_frontend(tmp_path, link_inputs=texts)["link"]


def resolve_citation_images(tmp_path, *pairs):
    return run_frontend(tmp_path, citation_inputs=pairs)["citation"]


def assert_no_raw_markup(output: str) -> None:
    """剔掉渲染器自己生成的标签后，不该还剩任何可当标记解释的字符。"""
    leftover = ALLOWED_TAG.sub("", output)
    for char in "<>\"":
        assert char not in leftover, f"输出里残留裸 {char!r}：{output}"


# ---------- 渲染安全 ----------

def test_script_tag_is_escaped(tmp_path):
    (output,) = render(tmp_path, "**<script>alert(1)</script>**")

    assert "<script" not in output
    assert "&lt;script&gt;" in output
    assert "<strong>" in output, "粗体仍应生效"
    assert_no_raw_markup(output)


def test_event_handler_attribute_is_escaped(tmp_path):
    (output,) = render(tmp_path, '<img src=x onerror="alert(1)">')

    assert "<img" not in output
    assert_no_raw_markup(output)


def test_quote_cannot_break_out_of_attribute(tmp_path):
    (output,) = render(tmp_path, '" onmouseover="alert(1)')

    assert "&quot;" in output
    assert_no_raw_markup(output)


def test_non_http_link_keeps_text_only(tmp_path):
    """不成链接的，只留文字——原样回显 [文字](url) 等于把源码摆给用户看。"""
    (output,) = render(tmp_path, "[点我](javascript:alert(1))")

    assert output == "<p>点我</p>"
    assert "javascript:" not in output
    assert ")(" not in output


def test_escape_covers_five_characters(tmp_path):
    (result,) = run_frontend(tmp_path, escape_inputs=['&<>"\''])["escape"]

    assert result == "&amp;&lt;&gt;&quot;&#39;"


# ---------- 渲染正确性 ----------

def test_bold_italic_and_inline_code(tmp_path):
    (output,) = render(tmp_path, "**粗** 与 *斜* 与 `code`")

    assert output == "<p><strong>粗</strong> 与 <em>斜</em> 与 <code>code</code></p>"


def test_bare_asterisks_stay_literal(tmp_path):
    (output,) = render(tmp_path, "2 * 3 * 4")

    assert output == "<p>2 * 3 * 4</p>"


def test_citation_becomes_anchor(tmp_path):
    (output,) = render(tmp_path, "结论见 [1] 与 [12]，不是 [abc] 也不是 [2024]")

    assert '<a class="cite" href="#cite-1">[1]</a>' in output
    assert '<a class="cite" href="#cite-12">[12]</a>' in output
    assert "[abc]" in output and "[2024]" in output


def test_citation_inside_code_is_not_linkified(tmp_path):
    inline, fenced = render(tmp_path, "`[1]`", "```\n[1]\n```")

    assert inline == "<p><code>[1]</code></p>"
    assert fenced == "<pre><code>[1]</code></pre>"


def test_blockquote_survives_escaping(tmp_path):
    (output,) = render(tmp_path, "> 引用一行\n> 接着一行")

    assert output == "<blockquote>引用一行接着一行</blockquote>"


def test_headings_start_at_h2(tmp_path):
    (output,) = render(tmp_path, "## 小节\n### 更小")

    assert output == "<h3>小节</h3>\n<h4>更小</h4>"


def test_lists_split_by_kind(tmp_path):
    (output,) = render(tmp_path, "- 甲\n- 乙\n\n1. 一\n2. 二")

    assert output == "<ul><li>甲</li><li>乙</li></ul>\n<ol><li>一</li><li>二</li></ol>"


def test_paragraph_breaks(tmp_path):
    (output,) = render(tmp_path, "第一段\n\n第二段")

    assert output == "<p>第一段</p>\n<p>第二段</p>"


def test_code_block_preserves_content(tmp_path):
    (output,) = render(tmp_path, "```python\nx = 1 < 2\n```")

    assert output == "<pre><code>x = 1 &lt; 2</code></pre>"


def test_empty_input_is_safe(tmp_path):
    assert render(tmp_path, "", "   ") == ["", ""]


def test_cjk_lines_join_without_space_latin_with(tmp_path):
    cjk, latin = render(tmp_path, "上文\n下文", "abc\ndef")

    assert cjk == "<p>上文下文</p>"
    assert latin == "<p>abc def</p>"


# ---------- 表格 ----------

# 用户实际提问时模型返回的表格
REAL_TABLE = """| 数据库 | 特点 | 适用场景 |
|---|---|---|
| Chroma | 上手最简单；但**没有分布式能力** | 实验、跑 Demo |
| Milvus | 功能最全、支持分布式部署 | 数据量百万到十亿级别 |"""


def test_table_renders_structure(tmp_path):
    (output,) = render(tmp_path, REAL_TABLE)

    assert output.startswith('<div class="table-wrap"><table>')
    assert output.count("<table>") == 1
    assert "<thead>" in output and "<tbody>" in output
    assert output.count("<th>") == 3
    assert output.count("<tr>") == 3  # 1 表头 + 2 数据行
    assert output.count("<td>") == 6
    assert_no_raw_markup(output)


def test_table_cell_inline_markup(tmp_path):
    (output,) = render(tmp_path, REAL_TABLE)

    assert "<strong>没有分布式能力</strong>" in output


def test_table_alignment(tmp_path):
    (output,) = render(tmp_path, "| 左 | 中 | 右 |\n|:---|:---:|---:|\n| a | b | c |")

    assert '<th style="text-align:left">左</th>' in output
    assert '<th style="text-align:center">中</th>' in output
    assert '<th style="text-align:right">右</th>' in output


def test_table_without_alignment_has_no_style(tmp_path):
    (output,) = render(tmp_path, "| a |\n|---|\n| b |")

    assert "<th>a</th>" in output
    assert "style=" not in output


def test_single_pipe_line_is_not_a_table(tmp_path):
    (output,) = render(tmp_path, "这个表格 | 只有一行竖线")

    assert "<table>" not in output
    assert output.startswith("<p>")


def test_table_requires_divider_row(tmp_path):
    (output,) = render(tmp_path, "| 头 | 值 |\n| 甲 | 乙 |")

    assert "<table>" not in output


def test_table_inside_code_fence_is_literal(tmp_path):
    (output,) = render(tmp_path, "```\n| a | b |\n|---|---|\n| 1 | 2 |\n```")

    assert "<table>" not in output
    assert "<pre><code>" in output


def test_table_survives_surrounding_paragraphs(tmp_path):
    (output,) = render(tmp_path, "上文\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n下文")

    assert output.startswith("<p>上文</p>")
    assert output.endswith("<p>下文</p>")
    assert "<table>" in output


def test_table_cells_are_escaped(tmp_path):
    (output,) = render(tmp_path, "| <script>x</script> |\n|---|\n| a |")

    assert "<script" not in output
    assert "&lt;script&gt;" in output
    assert_no_raw_markup(output)


# ---------- 链接 ----------

def test_http_link_renders_anchor(tmp_path):
    (output,) = render(tmp_path, "[文档](https://example.com/a)")

    assert output == (
        '<p><a href="https://example.com/a" target="_blank" '
        'rel="noopener noreferrer">文档</a></p>'
    )


def test_relative_link_absolutised_with_resolver(tmp_path):
    (output,) = render_with_link_resolver(tmp_path, "[1. 什么是 RAG？](/ai/rag/1_whatisrag.html)")

    assert output == (
        '<p><a href="https://xiaolinnote.com/ai/rag/1_whatisrag.html" target="_blank" '
        'rel="noopener noreferrer">1. 什么是 RAG？</a></p>'
    )


def test_relative_link_without_resolver_keeps_text(tmp_path):
    (output,) = render(tmp_path, "[文档](/ai/rag/1.html)")

    assert output == "<p>文档</p>"


def test_javascript_url_with_resolver_still_not_a_link(tmp_path):
    (output,) = render_with_link_resolver(tmp_path, "[点我](javascript:alert(1))")

    assert output == "<p>点我</p>"  # 不留孤立的 )
    assert "<a href" not in output


def test_link_and_citation_do_not_collide(tmp_path):
    (output,) = render_with_link_resolver(tmp_path, "[文档](/ai/rag/a.html) 与 [2]")

    assert '<a href="https://xiaolinnote.com/ai/rag/a.html"' in output
    assert '<a class="cite" href="#cite-2">[2]</a>' in output


def test_link_with_parentheses_in_url(tmp_path):
    (output,) = render(tmp_path, "[函数](https://example.com/f(x).html)")

    assert output == (
        '<p><a href="https://example.com/f(x).html" target="_blank" '
        'rel="noopener noreferrer">函数</a></p>'
    )


# ---------- 删除线 ----------

def test_strikethrough(tmp_path):
    (output,) = render(tmp_path, "~~作废~~ 保留")

    assert output == "<p><del>作废</del> 保留</p>"


def test_strikethrough_and_italic_do_not_collide(tmp_path):
    (output,) = render(tmp_path, "~~甲~~ 与 *乙*")

    assert output == "<p><del>甲</del> 与 <em>乙</em></p>"


# ---------- 嵌套列表 ----------

def test_nested_list(tmp_path):
    (output,) = render(tmp_path, "- 甲\n  - 甲一\n- 乙")

    assert output == "<ul><li>甲<ul><li>甲一</li></ul></li><li>乙</li></ul>"


def test_deeply_nested_list(tmp_path):
    (output,) = render(tmp_path, "- 一\n  - 二\n    - 三")

    assert output == "<ul><li>一<ul><li>二<ul><li>三</li></ul></li></ul></li></ul>"


def test_list_level_switches_between_kinds(tmp_path):
    (output,) = render(tmp_path, "- 甲\n1. 乙")

    assert output == "<ul><li>甲</li></ul><ol><li>乙</li></ol>"


def test_list_inline_markup_inside_items(tmp_path):
    (output,) = render(tmp_path, "- **粗** 项")

    assert output == "<ul><li><strong>粗</strong> 项</li></ul>"


# ---------- 图片 ----------

def test_image_renders_with_resolver(tmp_path):
    (output,) = render_with_image_resolver(tmp_path, "![向量库对比](assets/demo/01.png)")

    assert output == (
        '<p><img src="/kb/rag/assets/demo/01.png" alt="向量库对比" '
        'loading="lazy" decoding="async"></p>'
    )


def test_image_without_resolver_degrades_to_text(tmp_path):
    (output,) = render(tmp_path, "![](assets/demo/01.png)")

    assert output == "<p>[图片：01.png]</p>"


def test_image_absolute_url_is_not_served(tmp_path):
    """只有语料里的 assets/ 相对引用能解析；绝对地址无从判断归属，降级。"""
    (output,) = render_with_image_resolver(tmp_path, "![图](https://example.com/a.png)")

    assert output == "<p>[图片：图]</p>"


def test_image_inside_code_block_is_literal(tmp_path):
    (output,) = render_with_image_resolver(tmp_path, "```\n![a](assets/demo/01.png)\n```")

    assert "<img" not in output
    assert "<pre><code>" in output


def test_image_alt_is_escaped(tmp_path):
    (output,) = render_with_image_resolver(tmp_path, '![<script>x</script>](assets/demo/01.png)')

    assert "<script" not in output
    assert_no_raw_markup(output)


# ---------- 引用图片地址 ----------

def test_citation_image_url_for_two_level_site(tmp_path):
    """两层站点：文章标识是 <站点>/<栏目>/<文章>。"""
    (url,) = resolve_citation_images(
        tmp_path, ("xiaolinnote/rag/4_chunking", "assets/4_chunking/01.png")
    )

    assert url == "/kb/xiaolinnote/rag/assets/4_chunking/01.png"


def test_citation_image_url_for_deep_site(tmp_path):
    """深层站点：文章标识还多一层章节，基准跟着文章走而不是固定切两刀。"""
    (url,) = resolve_citation_images(
        tmp_path,
        ("xiaolincoding/os/1_hardware/how_cpu_run", "assets/how_cpu_run/08.png"),
    )

    assert url == "/kb/xiaolincoding/os/1_hardware/assets/how_cpu_run/08.png"


def test_citation_image_url_rejects_non_assets_reference(tmp_path):
    results = resolve_citation_images(
        tmp_path,
        ("xiaolinnote/rag/4_chunking", "https://example.com/a.png"),
        ("xiaolinnote/rag/4_chunking", "images/a.png"),
    )

    assert results == [None, None]


def test_citation_image_url_without_page_id_is_null(tmp_path):
    results = resolve_citation_images(
        tmp_path,
        ("", "assets/demo/01.png"),
        ("no-slash", "assets/demo/01.png"),
    )

    assert results == [None, None]

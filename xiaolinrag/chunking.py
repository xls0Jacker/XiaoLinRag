"""父子切分：把一篇文章切成「子块（检索定位）+ 父块（上下文阅读）」。

对齐 rag/4_chunking.md 的策略：
- 语义边界切割：顺着标题层级与段落切，不在语义中间截断
- 特殊内容专项处理：代码块、表格、列表整块保留，不截断
- 父子切割：小块检索命中，大块返回给模型阅读
段落自身超过子块上限时，按句子边界回退切分（笔记中的分隔符优先级思路）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from markdown_it import MarkdownIt

from .config import ChunkingConfig
from .corpus import Page

_MD = MarkdownIt("commonmark").enable(["table", "strikethrough"])

# 整块保留、不参与截断的块类型
_ATOMIC_KINDS = {"code", "table", "list"}

# 容器类 token：整体按源码切片取出，内部 token 不再单独处理
_CONTAINERS = {
    "bullet_list_open": "bullet_list_close",
    "ordered_list_open": "ordered_list_close",
    "table_open": "table_close",
    "blockquote_open": "blockquote_close",
}

_SENTENCE_END = set("。！？!?；;")


@dataclass
class Block:
    """正文中的一个语义块。"""

    kind: str  # heading | para | code | list | table | quote
    text: str
    level: int = 0  # 仅 heading 使用


@dataclass
class ChildChunk:
    """子块：向量化的检索定位单元。"""

    chunk_id: str  # "{page_id}#c{seq}"
    page_id: str
    heading_path: str  # 所属标题路径，如 "4. RAG… > 粒度怎么定"
    child_text: str
    parent_id: str
    seq: int
    context_note: str = ""  # Contextual Retrieval 生成的背景，由 contextual.py 填充


@dataclass
class ParentChunk:
    """父块：返回给生成模型阅读的上下文单元。"""

    parent_id: str  # "{page_id}#p{seq}"
    page_id: str
    parent_text: str
    child_ids: list[str] = field(default_factory=list)


def _find_close(tokens: list, start: int, open_type: str, close_type: str) -> int:
    """找到与 start 处容器 token 配对的结束 token 下标（支持嵌套）。"""
    depth = 0
    for j in range(start, len(tokens)):
        if tokens[j].type == open_type:
            depth += 1
        elif tokens[j].type == close_type:
            depth -= 1
            if depth == 0:
                return j
    return len(tokens) - 1


def _source_slice(lines: list[str], token) -> str:
    """按 token 的源码行范围取出原始 markdown（保留列表符号、表格竖线等结构）。"""
    if not token.map:
        return ""
    start, end = token.map
    return "\n".join(lines[start:end]).strip()


def _blocks(text: str) -> list[Block]:
    """把 markdown 正文解析成有序的语义块序列。"""
    tokens = _MD.parse(text)
    lines = text.splitlines()
    blocks: list[Block] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.type == "heading_open":
            content = tokens[i + 1].content.strip() if i + 1 < len(tokens) else ""
            if content:
                blocks.append(Block("heading", content, level=int(token.tag[1])))
            i += 3
            continue

        if token.type in _CONTAINERS:
            close_type = _CONTAINERS[token.type]
            end = _find_close(tokens, i, token.type, close_type)
            source = _source_slice(lines, token)
            if source:
                kind = "table" if token.type == "table_open" else (
                    "quote" if token.type == "blockquote_open" else "list"
                )
                blocks.append(Block(kind, source))
            i = end + 1
            continue

        if token.type in ("fence", "code_block"):
            lang = (token.info or "").strip().split(" ")[0] if token.info else ""
            body = token.content.rstrip("\n")
            if body.strip():
                blocks.append(Block("code", f"```{lang}\n{body}\n```"))
            i += 1
            continue

        if token.type == "paragraph_open":
            content = tokens[i + 1].content.strip() if i + 1 < len(tokens) else ""
            if content:
                blocks.append(Block("para", content))
            i += 3
            continue

        i += 1
    return blocks


def _groups(text: str) -> list[tuple[str, list[Block]]]:
    """按标题层级把语义块分组，返回 (标题路径, 该组内容块) 序列。"""
    groups: list[tuple[str, list[Block]]] = []
    stack: list[tuple[int, str]] = []
    path = ""
    current: list[Block] = []

    for block in _blocks(text):
        if block.kind == "heading":
            if current:
                groups.append((path, current))
                current = []
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.text))
            path = " > ".join(title for _, title in stack)
        else:
            current.append(block)

    if current:
        groups.append((path, current))
    return groups


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _SENTENCE_END:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def _split_by_sentence(text: str, max_len: int) -> list[str]:
    """超长段落按句子边界回退切分，极端情况下才硬切。"""
    pieces: list[str] = []
    for sentence in _sentences(text):
        while len(sentence) > max_len:
            pieces.append(sentence[:max_len])
            sentence = sentence[max_len:]
        pieces.append(sentence)

    merged: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > max_len:
            merged.append(current)
            current = ""
        current += piece
    if current:
        merged.append(current)
    return [stripped for stripped in (m.strip() for m in merged) if stripped]


def _block_pieces(block: Block, max_len: int) -> list[str]:
    """一个语义块贡献的片段：整块类永不截断，超长段落按句子回退。"""
    if block.kind in _ATOMIC_KINDS or len(block.text) <= max_len:
        return [block.text]
    return _split_by_sentence(block.text, max_len)


def _heading_prefix(heading: str, cfg: ChunkingConfig) -> str:
    """子块开头的标题前缀，长度受控，避免挤占正文预算。"""
    if not heading:
        return ""
    limit = max(cfg.child_max // 4, 1)
    title = heading if len(heading) + 2 <= limit else heading[: max(limit - 2, 1)]
    return f"{title}\n\n"


def _child_texts(blocks: list[Block], heading: str, cfg: ChunkingConfig) -> list[str]:
    """把一组语义块打包成子块文本，每块以最近标题开头以便自解释。

    标题前缀计入子块长度预算，保证含前缀后仍不超过上限。
    """
    prefix = _heading_prefix(heading, cfg)
    budget_max = max(cfg.child_max - len(prefix), 1)
    budget_target = min(max(cfg.child_target - len(prefix), 1), budget_max)

    bodies: list[str] = []
    buffer: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        if buffer:
            bodies.append("\n\n".join(buffer))
            buffer = []
            size = 0

    for block in blocks:
        for piece in _block_pieces(block, budget_max):
            length = len(piece)
            if length > budget_max:
                # 整块内容超过上限（如长代码块），单独成块，不截断
                flush()
                bodies.append(piece)
                continue
            if buffer and size + length + 2 > budget_max:
                flush()
            buffer.append(piece)
            size += length + 2
            if size >= budget_target:
                flush()
    flush()

    return [f"{prefix}{body}" for body in bodies]


def _parent_layout(child_texts: list[str], cfg: ChunkingConfig) -> list[list[int]]:
    """按顺序把子块打包成父块，返回每个父块包含的子块下标。"""
    layout: list[list[int]] = []
    current: list[int] = []
    size = 0
    for index, text in enumerate(child_texts):
        length = len(text)
        if current and (size + length > cfg.parent_max or size >= cfg.parent_target):
            layout.append(current)
            current = []
            size = 0
        current.append(index)
        size += length + 2
    if current:
        layout.append(current)
    return layout


def split_page(page: Page, cfg: ChunkingConfig) -> tuple[list[ChildChunk], list[ParentChunk]]:
    """把一篇文章切成子块与父块，父子通过 ID 关联。"""
    entries: list[tuple[str, str]] = []  # (标题路径, 子块文本)
    for path, blocks in _groups(page.text):
        heading = path.split(" > ")[-1] if path else ""
        for text in _child_texts(blocks, heading, cfg):
            entries.append((path, text))

    if not entries:
        return [], []

    children = [
        ChildChunk(
            chunk_id=f"{page.page_id}#c{seq}",
            page_id=page.page_id,
            heading_path=path,
            child_text=text,
            parent_id="",
            seq=seq,
        )
        for seq, (path, text) in enumerate(entries)
    ]

    parents: list[ParentChunk] = []
    for pseq, indexes in enumerate(_parent_layout([t for _, t in entries], cfg)):
        parent_id = f"{page.page_id}#p{pseq}"
        parents.append(
            ParentChunk(
                parent_id=parent_id,
                page_id=page.page_id,
                parent_text="\n\n".join(entries[i][1] for i in indexes),
                child_ids=[children[i].chunk_id for i in indexes],
            )
        )
        for i in indexes:
            children[i].parent_id = parent_id

    return children, parents

"""提示词模板：资料编号约定 [n] 是答案与引用列表对齐的唯一依据。

栏目与站点的展示名一律由调用方传入（来自索引元信息），本模块不再内置任何栏目名——
它们是知识库的数据，不是代码的常量。
"""

from __future__ import annotations

from .index_store import IndexEntry

SYSTEM_PROMPT = """你是「小林面试笔记」与「小林coding」两个知识库的学习助手，帮助用户准备技术面试——既覆盖 AI 应用开发方向（RAG、LLM、工具调用、LangChain、Agent、Claude Code），也覆盖计算机基础方向（网络、操作系统、MySQL、Redis）。

回答要求：
1. 只依据【参考资料】作答。不要引入资料之外的知识，也不要凭自己的印象补充细节。
2. 用中文回答，先给结论再展开要点，条理清晰，必要时用短列表。
3. 引用资料里的结论时，在句末标上对应编号，如 [1]、[2]；编号必须来自参考资料，不要编造。
4. 如果参考资料不足以回答问题，直接回答「知识库中未找到相关内容」，并说明还缺什么，不要编造答案。
5. 直接回答问题本身，不要复述这些要求。"""

USER_TEMPLATE = """【参考资料】
{context}

【用户问题】
{question}"""

# 出处行只写栏目展示名，不叠站点名：实测 10 个栏目的展示名两两不重复（清单见
# docs/05-多站点知识库.md），再叠一层是冗余。原先的「{section}栏目」写法也要去掉——
# 展示名本身就带「图解」这类词，后面再缀「栏目」二字会读成重复。
SOURCE_TEMPLATE = "[{index}] 《{title}》（{section} · {heading_path}）\n{text}"""


def section_label(key: str, labels: dict[str, str] | None = None) -> str:
    """栏目 key 转展示名。

    labels 是「复合栏目键 -> 展示名」的映射，来自索引元信息。查不到时兜底取
    复合键的最后一段（`xiaolincoding/network` → `network`），而不是留空——
    索引很旧、元信息里没有展示名时，界面上至少要看得出是哪一块。
    """
    if labels and key in labels:
        return labels[key]
    return key.rsplit("/", 1)[-1] if key else key


def build_context(entries: list[IndexEntry], labels: dict[str, str] | None = None) -> str:
    """把进入生成的父块组装成带编号的参考资料。"""
    blocks = [
        SOURCE_TEMPLATE.format(
            index=i,
            title=entry.title,
            section=section_label(entry.section, labels),
            heading_path=entry.heading_path or entry.title,
            text=entry.parent_text,
        )
        for i, entry in enumerate(entries, start=1)
    ]
    return "\n\n".join(blocks)


def build_user_prompt(
    question: str,
    entries: list[IndexEntry],
    labels: dict[str, str] | None = None,
) -> str:
    return USER_TEMPLATE.format(context=build_context(entries, labels), question=question.strip())

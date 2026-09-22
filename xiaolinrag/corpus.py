"""语料发现与解析：从知识库目录读取全部笔记页。

知识库目录由 XiaoLinCrawler 产出（本系统只读）：按站点分组，每个站点下若干栏目，
栏目目录下可再有章节子目录；每篇 .md 带 YAML frontmatter，另有 manifest.json
声明全量索引。发现数量与索引不一致时报错（N5）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class CorpusError(Exception):
    """语料缺失或与索引不一致，消息面向使用者可直接阅读。"""


@dataclass
class Page:
    page_id: str  # 相对知识库根去扩展名，如 "xiaolincoding/network/1_base/tcp_ip_model"
    site: str  # 站点 key，如 "xiaolincoding"
    section: str  # 复合栏目键，如 "xiaolincoding/network"
    title: str
    source_url: str
    text: str  # 去掉 frontmatter 的正文


@dataclass
class SectionInfo:
    """一个栏目的展示信息。key 是「站点/栏目」复合键，与 Page.section 同构。"""

    key: str
    site: str
    label: str
    page_count: int


@dataclass
class SiteInfo:
    """一个站点及其下属栏目。展示名取自知识库索引文件，不在源码里硬编码。"""

    key: str
    label: str
    sections: list[SectionInfo] = field(default_factory=list)


def _split_frontmatter(raw: str, path: Path) -> tuple[dict, str]:
    """拆出 YAML frontmatter 与正文。"""
    if not raw.startswith("---"):
        raise CorpusError(f"{path} 缺少 frontmatter（应以 --- 开头）")
    parts = raw.split("---", 2)
    if len(parts) < 3:
        raise CorpusError(f"{path} 的 frontmatter 未闭合（缺少结尾 ---）")
    meta = yaml.safe_load(parts[1]) or {}
    if not isinstance(meta, dict):
        raise CorpusError(f"{path} 的 frontmatter 不是键值对")
    return meta, parts[2].lstrip("\n")


def _read_manifest(kb: Path) -> dict:
    """读取知识库索引文件，校验它确实是当前爬虫版本的格式。"""
    manifest_path = kb / "manifest.json"
    if not manifest_path.exists():
        raise CorpusError(
            f"未找到知识库索引 {manifest_path}；请确认配置里的 kb_dir 指向 XiaoLinCrawler 的产物目录"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise CorpusError(f"{manifest_path} 不是合法 JSON：{exc}") from exc
    if not isinstance(manifest, dict) or not manifest.get("sites"):
        raise CorpusError(
            f"{manifest_path} 里没有 sites，无法确定抓取范围；"
            "请确认它由当前版本的爬虫产出（旧版索引只有顶层 sections）"
        )
    return manifest


def _iter_sections(manifest: dict) -> list[tuple[str, str]]:
    """展开索引声明的站点与栏目，返回 (站点 key, 栏目 key) 列表。"""
    plan: list[tuple[str, str]] = []
    for site in manifest["sites"]:
        site_key = site.get("key")
        sections = site.get("sections") or []
        if not site_key or not sections:
            raise CorpusError(
                f"索引中站点 {site_key or '（缺少 key）'} 没有声明栏目，无法确定抓取范围"
            )
        for section in sections:
            section_key = section.get("key")
            if not section_key:
                raise CorpusError(f"索引中站点 {site_key} 下有栏目缺少 key")
            plan.append((site_key, section_key))
    return plan


def _discover(kb: Path, manifest: dict) -> list[Path]:
    """扫描各站点栏目目录，并与索引声明的文件集合核对。

    栏目目录下允许再有章节子目录，所以用递归扫描；`assets/` 下的文件是图片资源，
    不是文章，即使混进了 .md 也不算语料。
    """
    found: list[Path] = []
    for site_key, section_key in _iter_sections(manifest):
        section_dir = kb / site_key / section_key
        if not section_dir.is_dir():
            raise CorpusError(f"索引声明的栏目目录不存在：{section_dir}")
        for path in section_dir.rglob("*.md"):
            if "assets" in path.relative_to(section_dir).parts:
                continue
            found.append(path)
    found.sort()

    expected = {p["md_path"] for p in manifest.get("pages", []) if p.get("md_path")}
    relative = {p.relative_to(kb).as_posix(): p for p in found}

    if expected:
        missing = sorted(expected - relative.keys())
        extra = sorted(relative.keys() - expected)
        if missing or extra:
            detail = []
            if missing:
                detail.append(f"索引里声明但磁盘上没有：{missing[:5]}")
            if extra:
                detail.append(f"磁盘上多出但索引未收录：{extra[:5]}")
            raise CorpusError(
                f"语料与索引不一致（索引 {len(expected)} 篇，磁盘 {len(relative)} 篇）；"
                + "；".join(detail)
            )
    else:
        declared = manifest.get("page_count")
        if declared and declared != len(found):
            raise CorpusError(
                f"语料数量与索引不一致：索引声明 {declared} 篇，磁盘发现 {len(found)} 篇"
            )

    return [relative[name] for name in sorted(relative)]


def load_pages(kb_dir: str | Path) -> list[Page]:
    """读取知识库全部笔记页，按相对路径排序返回。

    文章身份是它相对知识库根的路径（去扩展名），因此天然含站点前缀；
    栏目是「站点/栏目」复合键，两个站点将来出现同名栏目也不会串。
    """
    kb = Path(kb_dir)
    manifest = _read_manifest(kb)

    pages: list[Page] = []
    for path in _discover(kb, manifest):
        relative = path.relative_to(kb)
        meta, text = _split_frontmatter(path.read_text(encoding="utf-8"), path)
        title = meta.get("title")
        source_url = meta.get("source_url")
        if not title or not source_url:
            raise CorpusError(f"{path} 的 frontmatter 缺少 title 或 source_url")
        parts = relative.parts
        pages.append(
            Page(
                page_id=relative.with_suffix("").as_posix(),
                site=parts[0],
                section="/".join(parts[:2]),
                title=str(title),
                source_url=str(source_url),
                text=text,
            )
        )
    return pages


def load_sites(kb_dir: str | Path) -> list[SiteInfo]:
    """读取站点与栏目的展示信息（展示名、篇数），供索引元信息与界面使用。"""
    kb = Path(kb_dir)
    manifest = _read_manifest(kb)

    by_site: dict[str, dict] = {}
    for site in manifest["sites"]:
        site_key = str(site["key"])
        by_site[site_key] = {
            "label": str(site.get("name") or site_key),
            "sections": [
                SectionInfo(
                    key=f"{site_key}/{section['key']}",
                    site=site_key,
                    label=str(section.get("name") or section["key"]),
                    page_count=int(section.get("page_count") or 0),
                )
                for section in site.get("sections") or []
            ],
        }

    # 先按索引里的声明顺序建骨架，再补齐磁盘上确实存在的栏目——顺序稳定，
    # 界面上的分组顺序就不会随扫描顺序抖动。
    return [
        SiteInfo(key=site_key, label=info["label"], sections=info["sections"])
        for site_key, info in by_site.items()
    ]

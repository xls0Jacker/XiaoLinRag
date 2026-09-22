"""命令行入口：build / status / serve / eval。"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .chunking import split_page
from .config import Config, ConfigError, load_config
from .contextual import enrich
from .corpus import CorpusError, load_pages, load_sites
from .embedder import EmbeddingError, embed_documents
from .index_store import IndexEntry, IndexStore, IndexStoreError, build_meta
from .llm import LLMError
from .prompts import section_label
from .reranker import RerankError
from .reuse import ReuseSource

FAILURES = (
    ConfigError,
    CorpusError,
    IndexStoreError,
    EmbeddingError,
    LLMError,
    RerankError,
)


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # 屏蔽底层 HTTP 客户端与分词器的逐条日志，只保留本项目自己的进度与告警
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "urllib3", "jieba"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _dir_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{total / 1024 / 1024:.1f} MB"


def _site_summary(store: IndexStore) -> list[str]:
    """站点及其栏目数。展示名取自索引元信息，元信息没有就退回站点 key。"""
    labels = store.meta.get("site_labels") or {}
    counts: dict[str, int] = {}
    for section in store.sections:
        site = section.split("/", 1)[0]
        counts[site] = counts.get(site, 0) + 1
    return [f"{labels.get(site, site)}（{count} 个栏目）" for site, count in counts.items()]


def _section_summary(store: IndexStore) -> list[str]:
    labels = store.meta.get("section_labels") or {}
    return [section_label(section, labels) for section in store.sections]


def _build_entries(pages, cfg: Config):
    """切分全部文章，返回（子块, 父块, 文章索引）。"""
    children, parents = [], []
    for page in pages:
        page_children, page_parents = split_page(page, cfg.chunking)
        children.extend(page_children)
        parents.extend(page_parents)
    return children, parents


def _embed_entries(entries: list[IndexEntry], cfg: Config, reuse: ReuseSource) -> np.ndarray:
    """向量化，尽量复用上一版：只对未命中的条目发起调用，再按原行序拼成完整矩阵。

    索引产物仍是整体重建（三路行号照样对齐），这里省掉的只是对外部服务的调用。
    """
    if not entries:
        # 让 IndexStore.build 去报那句可读的「没有可索引的子块」，别在这里先抛 StopIteration
        return np.zeros((0, 0), dtype="float32")

    cached: dict[int, np.ndarray] = {}
    missed: list[int] = []
    for position, entry in enumerate(entries):
        vector = reuse.vector(reuse.vector_key(entry.embed_text))
        if vector is None:
            missed.append(position)
        else:
            cached[position] = vector

    if missed:
        fresh = embed_documents(
            [entries[position].embed_text for position in missed], cfg, on_progress=_progress
        )
        for slot, position in enumerate(missed):
            cached[position] = fresh[slot]

    dim = int(next(iter(cached.values())).shape[0])
    matrix = np.zeros((len(entries), dim), dtype="float32")
    for position, vector in cached.items():
        matrix[position] = vector
    return matrix


def _progress(done: int, total: int) -> None:
    print(f"  向量化进度 {done}/{total}", end="\r", flush=True)


def cmd_build(args) -> int:
    cfg = load_config(args.config)
    started = time.time()

    pages = load_pages(cfg.kb_dir)
    sites = load_sites(cfg.kb_dir)
    print(f"语料：{len(pages)} 篇，{len(sites)} 个站点")

    reuse = ReuseSource.load(cfg.index_dir, cfg)
    print(f"复用上一版索引：{reuse.reason}")

    children, parents = _build_entries(pages, cfg)
    print(f"切分：子块 {len(children)} 个，父块 {len(parents)} 个（耗时 {time.time() - started:.1f}s）")

    if cfg.contextual.enabled:
        print(f"生成背景说明（Contextual Retrieval，并发 {cfg.contextual.concurrency}）...")
        phase = time.time()
        enrich(children, {page.page_id: page for page in pages}, cfg, reuse=reuse)
        missing = sum(1 for child in children if not child.context_note)
        print(
            f"背景完成：{len(children) - missing}/{len(children)} 个"
            f"（缺 {missing} 个，耗时 {time.time() - phase:.1f}s）"
        )
        stats = reuse.stats
        print(
            f"背景复用：命中 {stats.context_pages_hit} 篇（{stats.context_chunks_hit} 个子块）"
            f"· 重新生成 {stats.context_pages_missed} 篇"
            f"· 省去 {stats.context_calls_saved} 次模型调用"
        )

    parent_text = {parent.parent_id: parent.parent_text for parent in parents}
    page_by_id = {page.page_id: page for page in pages}
    entries = [
        IndexEntry(
            chunk_id=child.chunk_id,
            parent_id=child.parent_id,
            page_id=child.page_id,
            section=page_by_id[child.page_id].section,
            title=page_by_id[child.page_id].title,
            source_url=page_by_id[child.page_id].source_url,
            heading_path=child.heading_path,
            child_text=child.child_text,
            context_note=child.context_note,
            parent_text=parent_text[child.parent_id],
            seq=child.seq,
        )
        for child in children
    ]

    print(f"向量化 {len(entries)} 个子块（{cfg.models.embedding_model}）...")
    phase = time.time()
    vectors = _embed_entries(entries, cfg, reuse)
    print(
        f"\n向量完成：{vectors.shape[0]} × {vectors.shape[1]} 维（耗时 {time.time() - phase:.1f}s）"
        f"；复用命中 {reuse.stats.vector_hit}，新向量化 {reuse.stats.vector_missed}"
    )

    meta = build_meta(
        embedding_model=cfg.models.embedding_model,
        chat_model=cfg.models.chat_model,
        page_count=len(pages),
        parent_count=len(parents),
        chunking=cfg.chunking,
        contextual=cfg.contextual,
        site_labels={site.key: site.label for site in sites},
        section_labels={
            section.key: section.label for site in sites for section in site.sections
        },
    )
    store = IndexStore.build(entries, vectors, meta)
    store.save(cfg.index_dir)

    print(
        f"索引已写入 {cfg.index_dir}（{_dir_size(cfg.index_dir)}）\n"
        f"总耗时 {time.time() - started:.1f}s"
    )
    return 0


def cmd_status(args) -> int:
    cfg = load_config(args.config)
    print(f"知识库目录：{cfg.kb_dir}")
    print(f"索引目录：{cfg.index_dir}")

    if not IndexStore.exists(cfg.index_dir):
        print("索引状态：尚未构建，请先运行 `xiaolinrag build`")
        return 1

    store = IndexStore.load(cfg.index_dir)
    print("索引状态：已就绪")
    print(f"  构建时间：{store.meta.get('built_at', '未知')}")
    print(f"  向量模型：{store.meta.get('embedding_model', '未知')}（{store.meta.get('dim')} 维）")
    print(f"  对话模型：{store.meta.get('chat_model', '未记录（旧版索引）')}")
    print(f"  文章数：{store.meta.get('page_count')}  父块数：{store.meta.get('parent_count')}")
    print(f"  子块数：{len(store.entries)}")
    print(f"  站点：{', '.join(_site_summary(store))}")
    print(f"  栏目：{', '.join(_section_summary(store))}")
    print(f"  索引体积：{_dir_size(cfg.index_dir)}")
    print(f"  检索参数：{vars(cfg.retrieval)}")
    print(f"  门控阈值：{cfg.retrieval.gate_threshold}")
    return 0


def cmd_serve(args) -> int:
    cfg = load_config(args.config)
    from . import api

    api.serve(cfg)
    return 0


def cmd_eval(args) -> int:
    cfg = load_config(args.config)
    from .eval import EvalError, run_eval, format_report

    try:
        if not IndexStore.exists(cfg.index_dir):
            raise IndexStoreError(
                f"索引尚未构建（{cfg.index_dir}）；请先运行 `xiaolinrag build` 再做评估"
            )
        store = IndexStore.load(cfg.index_dir)
        report = run_eval(
            cfg,
            store,
            judge_sample=args.judge_sample,
            skip_judge=args.no_judge,
        )
    except EvalError as exc:
        raise IndexStoreError(f"评估无法进行：{exc}") from exc
    print(format_report(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xiaolinrag", description="小林面试笔记 RAG 问答系统")
    parser.add_argument("-c", "--config", default=None, help="配置文件路径（默认 config.yaml）")
    sub = parser.add_subparsers(dest="command")

    # 每个子命令也接受 -c，让 `build -c x.yaml` 与 `-c x.yaml build` 都能用。
    # default=SUPPRESS 很关键：否则子命令会把顶层已解析的值覆盖成 None。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-c",
        "--config",
        default=argparse.SUPPRESS,
        help="配置文件路径（默认 config.yaml）",
    )

    build = sub.add_parser("build", parents=[common], help="构建/重建索引")
    build.set_defaults(handler=cmd_build)

    status = sub.add_parser("status", parents=[common], help="查看索引状态")
    status.set_defaults(handler=cmd_status)

    serve = sub.add_parser("serve", parents=[common], help="启动 Web 问答界面")
    serve.set_defaults(handler=cmd_serve)

    evaluate = sub.add_parser(
        "eval", parents=[common], help="离线评估（检索层 Hit@K/MRR + 生成层 LLM 裁判）"
    )
    evaluate.add_argument("--judge-sample", type=int, default=10, help="生成层抽样题数（默认 10）")
    evaluate.add_argument("--no-judge", action="store_true", help="只跑检索层指标")
    evaluate.set_defaults(handler=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> None:
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return
    try:
        sys.exit(args.handler(args))
    except FAILURES as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)

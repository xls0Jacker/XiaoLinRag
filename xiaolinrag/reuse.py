"""增量复用：把上一版索引当作复用源，避免为没变过的内容重复付费。

做法是「按内容寻址」而不是按位置寻址：背景说明的键由文章的子块序列与切分参数算出，
向量的键由送入向量化的文本算出。上一版索引里本来就存着这两样东西，所以只要内容没变
就自然命中，内容变了就自然未命中——不需要另立一套缓存文件。

索引产物仍然整体重建，这里省掉的只是对外部服务的调用（见 spec 的「不做的事」）。

哪些情况算「内容没变」由元信息显式把关：上一版索引记录的对话模型/向量模型与当前配置
不一致时，对应的复用整批关闭。模型名不参与哈希键，这样同一模型换个 ID 写法不会误伤。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .config import Config
from .index_store import ENTRIES_FILE, INDEX_FILE, META_FILE

logger = logging.getLogger(__name__)

CONTEXT_PREFIX = b"ctx-v1\n"
VECTOR_PREFIX = b"vec-v1\n"


def _digest(prefix: bytes, parts: list[str]) -> str:
    """把若干文本拼进一个哈希：加长度前缀，避免相邻片段错位后撞键。"""
    hasher = hashlib.sha256()
    hasher.update(prefix)
    for part in parts:
        encoded = part.encode("utf-8")
        hasher.update(str(len(encoded)).encode("ascii"))
        hasher.update(b":")
        hasher.update(encoded)
        hasher.update(b"\n")
    return hasher.hexdigest()


class ReuseStats:
    """本次构建从上一版索引里省下了多少。"""

    def __init__(self) -> None:
        self.context_pages_hit = 0
        self.context_pages_missed = 0
        self.context_chunks_hit = 0
        self.context_calls_saved = 0
        self.vector_hit = 0
        self.vector_missed = 0

    def __repr__(self) -> str:  # pragma: no cover - 仅用于日志排查
        return (
            f"ReuseStats(context_pages_hit={self.context_pages_hit}, "
            f"context_pages_missed={self.context_pages_missed}, "
            f"context_chunks_hit={self.context_chunks_hit}, "
            f"context_calls_saved={self.context_calls_saved}, "
            f"vector_hit={self.vector_hit}, vector_missed={self.vector_missed})"
        )


class ReuseSource:
    """上一版索引的只读视图；无法使用时退化为「什么都不命中」的空源。"""

    def __init__(
        self,
        contexts: dict[str, list[str]],
        vectors: dict[str, np.ndarray],
        reason: str,
        loaded: bool = True,
    ) -> None:
        self._contexts = contexts
        self._vectors = vectors
        self._reason = reason
        self._loaded = loaded
        self.stats = ReuseStats()

    # ---------- 装配 ----------

    @classmethod
    def empty(cls, reason: str) -> "ReuseSource":
        """读不到上一版索引——什么都没得复用。"""
        return cls({}, {}, reason, loaded=False)

    @classmethod
    def load(cls, index_dir: str | Path, cfg: Config) -> "ReuseSource":
        """读上一版索引。任何异常都退化为空源，绝不让构建因此失败。"""
        target = Path(index_dir)
        missing = [
            name
            for name in (INDEX_FILE, ENTRIES_FILE, META_FILE)
            if not (target / name).exists()
        ]
        if missing:
            return cls.empty(f"{target} 下没有上一版索引（缺 {', '.join(missing)}），本次全量生成")

        try:
            meta = json.loads((target / META_FILE).read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in (target / ENTRIES_FILE).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, ValueError) as exc:
            return cls.empty(f"上一版索引读不出来（{exc}），本次全量生成")

        if not rows:
            return cls.empty("上一版索引里没有条目，本次全量生成")

        # 模型不一致就整批关掉：键里不含模型名，靠这里显式把关。
        contexts_usable = meta.get("chat_model") == cfg.models.chat_model
        vectors_usable = meta.get("embedding_model") == cfg.models.embedding_model
        notes: list[str] = []
        if not contexts_usable:
            notes.append(
                f"对话模型已从 {meta.get('chat_model') or '（未记录）'} 变为 "
                f"{cfg.models.chat_model}，背景说明需重新生成"
            )
        if not vectors_usable:
            notes.append(
                f"向量模型已从 {meta.get('embedding_model') or '（未记录）'} 变为 "
                f"{cfg.models.embedding_model}，向量需重新生成"
            )

        contexts = cls._collect_contexts(rows, cfg) if contexts_usable else {}
        vectors = cls._collect_vectors(rows, target) if vectors_usable else {}
        return cls(contexts, vectors, "；".join(notes) or "上一版索引可用")

    @staticmethod
    def _collect_contexts(rows: list[dict], cfg: Config) -> dict[str, list[str]]:
        """按文章聚合：键由该文按序的子块原文算出，值为按序的背景说明。"""
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("page_id", "")), []).append(row)

        collected: dict[str, list[str]] = {}
        for page_id, page_rows in grouped.items():
            page_rows.sort(key=lambda row: row.get("seq", 0))
            child_texts = [str(row.get("child_text", "")) for row in page_rows]
            collected[_context_key(child_texts, cfg)] = [
                str(row.get("context_note", "")) for row in page_rows
            ]
        return collected

    @staticmethod
    def _collect_vectors(rows: list[dict], target: Path) -> dict[str, np.ndarray]:
        """按「送入向量化的文本」聚合向量；行号与上一版索引一一对应。"""
        try:
            index = faiss.read_index(str(target / INDEX_FILE))
        except Exception as exc:  # noqa: BLE001 — 读不出来就当没有，不阻断构建
            logger.warning("上一版向量索引读不出来（%s），本次重新向量化", exc)
            return {}
        if index.ntotal != len(rows):
            logger.warning(
                "上一版索引的向量数（%s）与条目数（%s）对不上，本次重新向量化",
                index.ntotal,
                len(rows),
            )
            return {}

        collected: dict[str, np.ndarray] = {}
        for row_id, row in enumerate(rows):
            embed_text = _embed_text(row)
            if embed_text in collected:
                continue
            collected[_vector_key(embed_text)] = np.asarray(index.reconstruct(row_id), dtype=np.float32)
        return collected

    # ---------- 查询 ----------

    @property
    def available(self) -> bool:
        """上一版索引是否读进来了。模型变更不改变这个值——那是 reason 的事。"""
        return self._loaded

    @property
    def reason(self) -> str:
        return self._reason

    def context_key(self, child_texts: list[str], cfg: Config) -> str:
        return _context_key(child_texts, cfg)

    def contexts(self, key: str) -> list[str] | None:
        """取该文的背景说明；未命中或整篇都是空说明时返回 None（宁可重算）。"""
        notes = self._contexts.get(key)
        if notes is None or not any(notes):
            self.stats.context_pages_missed += 1
            return None
        self.stats.context_pages_hit += 1
        self.stats.context_chunks_hit += len(notes)
        return notes

    def vector_key(self, embed_text: str) -> str:
        return _vector_key(embed_text)

    def vector(self, key: str) -> np.ndarray | None:
        found = self._vectors.get(key)
        if found is None:
            self.stats.vector_missed += 1
        else:
            self.stats.vector_hit += 1
        return found


def _embed_text(row: dict) -> str:
    """与 IndexEntry.embed_text 同一条规则：有背景就「背景 + 原文」。"""
    context_note = str(row.get("context_note", ""))
    child_text = str(row.get("child_text", ""))
    return f"{context_note}\n\n{child_text}" if context_note else child_text


def _context_key(child_texts: list[str], cfg: Config) -> str:
    """背景说明的键：子块序列 + 切分参数（含背景字数上限）。"""
    chunking = json.dumps(asdict(cfg.chunking), sort_keys=True, ensure_ascii=False)
    return _digest(CONTEXT_PREFIX, [chunking, *child_texts])


def _vector_key(embed_text: str) -> str:
    """向量的键：送入向量化的文本本身。"""
    return _digest(VECTOR_PREFIX, [embed_text])

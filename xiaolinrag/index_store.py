"""索引存储与检索：FAISS 向量 + JSONL 条目 + BM25 关键词。

三路按同一行序对齐：FAISS 行号 = entries.jsonl 行号 = BM25 文档序，
由 build 一次性建立，load 时校验一致（不一致即提示重建）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

INDEX_FILE = "index.faiss"
ENTRIES_FILE = "entries.jsonl"
META_FILE = "meta.json"


class IndexStoreError(Exception):
    """索引缺失或损坏，消息面向使用者可直接阅读。"""


@dataclass
class IndexEntry:
    """索引里的一条子块记录，附带其父块原文与来源元数据。"""

    chunk_id: str
    parent_id: str
    page_id: str
    section: str
    title: str
    source_url: str
    heading_path: str
    child_text: str
    context_note: str
    parent_text: str
    seq: int

    @property
    def site(self) -> str:
        """站点 = 文章标识的首段。恒成立，故不落库——不加字段，旧条目文件才读得进来。"""
        return self.page_id.split("/", 1)[0]

    @property
    def embed_text(self) -> str:
        """送入 Embedding 的文本：Contextual 背景 + 子块原文。"""
        if self.context_note:
            return f"{self.context_note}\n\n{self.child_text}"
        return self.child_text


@dataclass
class ScoredHit:
    """一条带分数的命中。score 为通道分数或融合分，rerank_score 在精排后填写。"""

    entry: IndexEntry
    score: float
    rerank_score: float | None = None


def _tokenize(text: str) -> list[str]:
    """中文分词，去掉纯标点/空白，英文统一小写。"""
    tokens = []
    for token in jieba.lcut(text):
        stripped = token.strip().lower()
        if stripped and any(ch.isalnum() or "一" <= ch <= "鿿" for ch in stripped):
            tokens.append(stripped)
    return tokens


class IndexStore:
    """索引的内存表示，负责构建、落盘、加载与两条检索通道。"""

    def __init__(self, entries: list[IndexEntry], index, meta: dict) -> None:
        self.entries = entries
        self.index = index
        self.meta = meta
        self._bm25: BM25Okapi | None = None
        self._bm25_tokens: list[list[str]] = []

    # ---------- 存储 ----------

    @classmethod
    def build(cls, entries: list[IndexEntry], vectors: np.ndarray, meta: dict) -> "IndexStore":
        """用条目与向量建索引，校验数量与维度一致。"""
        if len(entries) == 0:
            raise IndexStoreError("没有可索引的子块，请检查语料与切分配置")
        if vectors.ndim != 2 or vectors.shape[0] != len(entries):
            raise IndexStoreError(
                f"向量数量（{vectors.shape[0] if vectors.ndim == 2 else '非法'}）"
                f"与条目数量（{len(entries)}）不一致"
            )
        dim = int(vectors.shape[1])
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        base = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap2(base)
        index.add_with_ids(vectors, np.arange(len(entries), dtype="int64"))

        meta = {**meta, "dim": dim, "child_count": len(entries)}
        return cls(entries, index, meta)

    def save(self, index_dir: str | Path) -> None:
        """落盘三件套：FAISS 索引、条目 JSONL、元信息。"""
        target = Path(index_dir)
        target.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(target / INDEX_FILE))
        with (target / ENTRIES_FILE).open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        (target / META_FILE).write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def exists(index_dir: str | Path) -> bool:
        target = Path(index_dir)
        return all((target / name).exists() for name in (INDEX_FILE, ENTRIES_FILE, META_FILE))

    @classmethod
    def load(cls, index_dir: str | Path) -> "IndexStore":
        """加载索引，校验文件齐全与行数对齐。"""
        target = Path(index_dir)
        missing = [
            name
            for name in (INDEX_FILE, ENTRIES_FILE, META_FILE)
            if not (target / name).exists()
        ]
        if missing:
            raise IndexStoreError(
                f"索引不完整（缺少 {', '.join(missing)}）：{target}；请先运行 `xiaolinrag build`"
            )

        entries = [
            IndexEntry(**json.loads(line))
            for line in (target / ENTRIES_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        meta = json.loads((target / META_FILE).read_text(encoding="utf-8"))
        index = faiss.read_index(str(target / INDEX_FILE))

        if index.ntotal != len(entries):
            raise IndexStoreError(
                f"索引与条目数量不一致（向量 {index.ntotal} 条，条目 {len(entries)} 条）："
                f"{target}；请重新运行 `xiaolinrag build`"
            )
        return cls(entries, index, meta)

    # ---------- 检索 ----------

    @property
    def sections(self) -> list[str]:
        """索引中出现的栏目 key，按字典序。"""
        return sorted({entry.section for entry in self.entries})

    def _allowed_rows(self, sections: list[str] | None) -> np.ndarray | None:
        if not sections:
            return None
        wanted = set(sections)
        rows = [i for i, entry in enumerate(self.entries) if entry.section in wanted]
        return np.asarray(rows, dtype="int64")

    def search_vector(
        self, vector: np.ndarray, top_k: int, sections: list[str] | None = None
    ) -> list[ScoredHit]:
        """向量通道：内积（向量已归一化，等价余弦相似度）。"""
        query = np.ascontiguousarray(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        expected = int(self.meta.get("dim", query.shape[1]))
        if query.shape[1] != expected:
            raise IndexStoreError(
                f"查询向量维度 {query.shape[1]} 与索引维度 {expected} 不一致；"
                f"可能是更换了 Embedding 模型，请重新运行 `xiaolinrag build`"
            )

        rows = self._allowed_rows(sections)
        if rows is not None and rows.size == 0:
            return []
        k = min(top_k, len(self.entries), int(rows.size) if rows is not None else len(self.entries))

        if rows is not None:
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(rows)
            scores, ids = self.index.search(query, k, params=params)
        else:
            scores, ids = self.index.search(query, k)

        return [
            ScoredHit(self.entries[int(i)], float(s))
            for s, i in zip(scores[0], ids[0])
            if i >= 0
        ]

    def _ensure_bm25(self) -> BM25Okapi:
        if self._bm25 is None:
            self._bm25_tokens = [_tokenize(entry.child_text) for entry in self.entries]
            self._bm25 = BM25Okapi(self._bm25_tokens)
        return self._bm25

    def search_bm25(
        self, query: str, top_k: int, sections: list[str] | None = None
    ) -> list[ScoredHit]:
        """关键词通道：BM25 over jieba 分词后的子块文本。"""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._ensure_bm25().get_scores(tokens), dtype=np.float32)

        rows = self._allowed_rows(sections)
        if rows is not None:
            if rows.size == 0:
                return []
            masked = np.full(scores.shape, -np.inf, dtype=np.float32)
            masked[rows] = scores[rows]
            scores = masked

        k = min(top_k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [ScoredHit(self.entries[int(i)], float(scores[i])) for i in top if scores[i] > 0]


def build_meta(
    embedding_model: str,
    chat_model: str,
    page_count: int,
    parent_count: int,
    chunking,
    contextual,
    site_labels: dict[str, str] | None = None,
    section_labels: dict[str, str] | None = None,
) -> dict:
    """构建时写入的元信息，供 status 展示与复现校验（N3）。

    `chat_model` 是背景说明的生成者，也是增量复用的判定依据之一——记录它，
    换模型后才有据可查（原先只有向量模型，对话模型无从判断）。
    """
    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": embedding_model,
        "chat_model": chat_model,
        "page_count": page_count,
        "parent_count": parent_count,
        "chunking": {
            "child_target": chunking.child_target,
            "child_max": chunking.child_max,
            "parent_target": chunking.parent_target,
            "parent_max": chunking.parent_max,
            "context_max_chars": chunking.context_max_chars,
        },
        "contextual_enabled": bool(contextual.enabled),
        "site_labels": dict(site_labels or {}),
        "section_labels": dict(section_labels or {}),
    }

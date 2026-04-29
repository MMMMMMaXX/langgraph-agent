"""SQLite knowledge catalog.

这里保存知识库文档和 chunk 的结构化真相源，同时维护 FTS5 lexical index。
Chroma 只负责 dense 向量召回；如果 Chroma 数据损坏，可以用这里的 chunk 重建。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.config import KNOWLEDGE_BASE_CONFIG
from app.retrieval.lexical.tokenizer import build_fts_index_text, build_fts_query

FTS_TABLE = "document_chunks_fts"


@dataclass(frozen=True)
class KnowledgeChunkRecord:
    """写入 knowledge catalog 的最小 chunk 记录。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    source: str
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    chunk_char_len: int
    section_title: str = ""
    metadata: dict | None = None


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _connect(path: str | Path | None = None) -> sqlite3.Connection:
    db_path = Path(path or KNOWLEDGE_BASE_CONFIG.sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def normalize_bm25_scores(rows: list[sqlite3.Row]) -> list[float]:
    """把 FTS5 bm25 分转换为越大越相关的 0~1 分数。"""

    if not rows:
        return []

    raw_scores = [float(row["bm25_score"]) for row in rows]
    best = min(raw_scores)
    worst = max(raw_scores)
    if best == worst:
        return [1.0 for _ in rows]

    return [(worst - score) / (worst - best) for score in raw_scores]


class KnowledgeCatalog:
    """知识库 SQLite catalog 的最小访问层。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or KNOWLEDGE_BASE_CONFIG.sqlite_path)

    def init_schema(self) -> None:
        with _connect(self.path) as conn:
            conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'unknown',
                    content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{{}}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    doc_title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    section_title TEXT NOT NULL DEFAULT '',
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    chunk_char_len INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{{}}',
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE}
                USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    doc_title,
                    section_title,
                    source UNINDEXED,
                    content,
                    tokenize='unicode61'
                );

                CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id
                ON document_chunks(doc_id);
                """
            )

    def reset(self) -> None:
        self.init_schema()
        with _connect(self.path) as conn:
            conn.execute(f"DELETE FROM {FTS_TABLE}")
            conn.execute("DELETE FROM document_chunks")
            conn.execute("DELETE FROM documents")

    def list_documents(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """分页读取已导入文档列表，供 API 和调试脚本使用。"""

        self.init_schema()
        with _connect(self.path) as conn:
            rows = conn.execute(
                """
                SELECT
                    d.doc_id,
                    d.title,
                    d.source,
                    d.source_type,
                    d.content_hash,
                    d.created_at,
                    d.updated_at,
                    COUNT(c.chunk_id) AS chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON c.doc_id = d.doc_id
                GROUP BY d.doc_id
                ORDER BY d.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (max(limit, 0), max(offset, 0)),
            ).fetchall()

        return [
            {
                "doc_id": row["doc_id"],
                "title": row["title"],
                "source": row["source"],
                "source_type": row["source_type"],
                "content_hash": row["content_hash"],
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
                "chunk_count": int(row["chunk_count"]),
            }
            for row in rows
        ]

    def get_document(self, doc_id: str) -> dict | None:
        """读取单个文档及其 chunk 摘要。"""

        self.init_schema()
        with _connect(self.path) as conn:
            row = conn.execute(
                """
                SELECT
                    doc_id, title, source, source_type, content_hash,
                    metadata_json, created_at, updated_at
                FROM documents
                WHERE doc_id = ?
                """,
                (doc_id,),
            ).fetchone()
            if row is None:
                return None

            chunk_rows = conn.execute(
                """
                SELECT
                    chunk_id, section_title, chunk_index, start_char,
                    end_char, chunk_char_len, content
                FROM document_chunks
                WHERE doc_id = ?
                ORDER BY chunk_index ASC
                """,
                (doc_id,),
            ).fetchall()

        return {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "source": row["source"],
            "source_type": row["source_type"],
            "content_hash": row["content_hash"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
            "chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "section_title": chunk["section_title"],
                    "chunk_index": int(chunk["chunk_index"]),
                    "start_char": int(chunk["start_char"]),
                    "end_char": int(chunk["end_char"]),
                    "chunk_char_len": int(chunk["chunk_char_len"]),
                    "preview": chunk["content"][:120],
                }
                for chunk in chunk_rows
            ],
        }

    def list_chunks(self, doc_id: str | None = None) -> list[dict]:
        """读取 catalog 中的 chunk 全量信息，用于重建 Chroma dense index。"""

        self.init_schema()
        where_sql = "WHERE doc_id = ?" if doc_id else ""
        params = (doc_id,) if doc_id else ()
        with _connect(self.path) as conn:
            rows = conn.execute(
                f"""
                SELECT
                    chunk_id, doc_id, doc_title, source, section_title,
                    chunk_index, content, start_char, end_char,
                    chunk_char_len, metadata_json
                FROM document_chunks
                {where_sql}
                ORDER BY doc_id ASC, chunk_index ASC
                """,
                params,
            ).fetchall()

        return [
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "doc_title": row["doc_title"],
                "source": row["source"],
                "section_title": row["section_title"],
                "chunk_index": int(row["chunk_index"]),
                "content": row["content"],
                "start_char": int(row["start_char"]),
                "end_char": int(row["end_char"]),
                "chunk_char_len": int(row["chunk_char_len"]),
                "metadata": json.loads(row["metadata_json"] or "{}"),
            }
            for row in rows
        ]

    def delete_document(self, doc_id: str) -> dict:
        """删除单篇文档及其 SQLite chunk / FTS5 记录。"""

        self.init_schema()
        with _connect(self.path) as conn:
            existing = conn.execute(
                "SELECT doc_id FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if existing is None:
                return {"deleted": False, "doc_id": doc_id, "chunk_count": 0}

            chunk_count = int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM document_chunks WHERE doc_id = ?",
                    (doc_id,),
                ).fetchone()["count"]
            )
            conn.execute(f"DELETE FROM {FTS_TABLE} WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

        return {"deleted": True, "doc_id": doc_id, "chunk_count": chunk_count}

    def upsert_document(
        self,
        *,
        doc_id: str,
        title: str,
        source: str,
        content: str,
        source_type: str = "json",
        metadata: dict | None = None,
    ) -> None:
        self.init_schema()
        now = time.time()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with _connect(self.path) as conn:
            existing = conn.execute(
                "SELECT created_at FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            created_at = float(existing["created_at"]) if existing else now
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    doc_id, title, source, source_type, content_hash,
                    metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    title,
                    source,
                    source_type,
                    content_hash(content),
                    metadata_json,
                    created_at,
                    now,
                ),
            )

    def replace_chunks(self, chunks: Iterable[KnowledgeChunkRecord]) -> int:
        self.init_schema()
        chunk_list = list(chunks)
        with _connect(self.path) as conn:
            doc_ids = sorted({chunk.doc_id for chunk in chunk_list})
            for doc_id in doc_ids:
                conn.execute(
                    f"DELETE FROM {FTS_TABLE} WHERE doc_id = ?",
                    (doc_id,),
                )
                conn.execute("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,))

            for chunk in chunk_list:
                metadata_json = json.dumps(chunk.metadata or {}, ensure_ascii=False)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO document_chunks (
                        chunk_id, doc_id, doc_title, source, section_title,
                        chunk_index, content, start_char, end_char,
                        chunk_char_len, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.doc_title,
                        chunk.source,
                        chunk.section_title,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.chunk_char_len,
                        metadata_json,
                    ),
                )
                conn.execute(
                    f"""
                    INSERT INTO {FTS_TABLE} (
                        chunk_id, doc_id, doc_title, section_title, source, content
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.doc_title,
                        chunk.section_title,
                        chunk.source,
                        build_fts_index_text(chunk.content),
                    ),
                )
        return len(chunk_list)

    def search_chunks(self, query: str, top_k: int) -> list[dict]:
        self.init_schema()
        match_query = build_fts_query(query)
        if not match_query:
            return []

        with _connect(self.path) as conn:
            rows = conn.execute(
                f"""
                SELECT
                    c.chunk_id,
                    c.doc_id,
                    c.doc_title,
                    c.source,
                    c.section_title,
                    c.chunk_index,
                    c.content,
                    c.start_char,
                    c.end_char,
                    c.chunk_char_len,
                    bm25({FTS_TABLE}) AS bm25_score
                FROM {FTS_TABLE}
                JOIN document_chunks c ON c.chunk_id = {FTS_TABLE}.chunk_id
                WHERE {FTS_TABLE} MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (match_query, max(top_k, 0)),
            ).fetchall()

        normalized_scores = normalize_bm25_scores(rows)
        hits: list[dict] = []
        for row, lexical_score in zip(rows, normalized_scores, strict=False):
            hits.append(
                {
                    "id": row["chunk_id"],
                    "content": row["content"],
                    "doc_id": str(row["doc_id"]),
                    "doc_title": str(row["doc_title"]),
                    "source": str(row["source"]),
                    "section_title": str(row["section_title"]),
                    "chunk_index": int(row["chunk_index"]),
                    "start_char": int(row["start_char"]),
                    "end_char": int(row["end_char"]),
                    "chunk_char_len": int(row["chunk_char_len"]),
                    "bm25_score": float(row["bm25_score"]),
                    "keyword_score": lexical_score,
                    "keyword_score_norm": lexical_score,
                }
            )
        return hits

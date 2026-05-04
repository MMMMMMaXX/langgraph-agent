"""知识库文档导入服务。

MVP 目标是先打通“文档内容 -> chunk -> SQLite FTS5 -> Chroma dense index”
闭环。这里不关心 HTTP 细节，API、脚本、后续文件上传都可以复用这一层。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.chunking import chunk_document_text
from app.config import VECTOR_STORE_CONFIG
from app.knowledge.catalog import (
    KnowledgeCatalog,
    KnowledgeChunkRecord,
    content_hash,
)
from app.llm import embed_texts
from app.vector_store import ChromaVectorStore

SUPPORTED_SOURCE_TYPES = {"txt", "md", "json"}
DEFAULT_BATCH_SIZE = 16
DOC_ID_HASH_CHARS = 16
TITLE_PREVIEW_CHARS = 40


@dataclass(frozen=True)
class KnowledgeImportInput:
    """导入服务的稳定输入模型。

    content 是文档正文；如果 source_type=json，content 可以是 JSON 字符串，
    服务会尝试从其中提取 title/source/content/metadata。
    """

    content: str
    doc_id: str = ""
    title: str = ""
    source: str = ""
    source_type: str = "txt"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeImportResult:
    """导入完成后的结果摘要。"""

    doc_id: str
    title: str
    source: str
    source_type: str
    content_hash: str
    content_char_len: int
    chunk_count: int
    indexed_to_sqlite: bool
    indexed_to_chroma: bool


def _normalize_source_type(source_type: str, source: str = "") -> str:
    explicit = source_type.strip().lower()
    if explicit in SUPPORTED_SOURCE_TYPES:
        return explicit

    suffix = Path(source).suffix.lower().lstrip(".")
    if suffix in SUPPORTED_SOURCE_TYPES:
        return suffix

    return "txt"


def _extract_markdown_title(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.lstrip("#").strip()
    return ""


def _infer_title(*, title: str, source: str, content: str) -> str:
    if title.strip():
        return title.strip()

    markdown_title = _extract_markdown_title(content)
    if markdown_title:
        return markdown_title

    if source:
        stem = Path(source).stem
        if stem:
            return stem

    preview = " ".join(content.split())[:TITLE_PREVIEW_CHARS]
    return preview or "untitled-document"


def _parse_json_document(raw_content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        return {"content": raw_content}

    if not isinstance(parsed, dict):
        return {"content": raw_content}

    content = str(
        parsed.get("content") or parsed.get("text") or parsed.get("body") or "",
    )
    metadata = (
        parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
    )
    return {
        "doc_id": str(parsed.get("doc_id") or parsed.get("id") or ""),
        "title": str(parsed.get("title") or parsed.get("name") or ""),
        "source": str(parsed.get("source") or parsed.get("path") or ""),
        "content": content or raw_content,
        "metadata": {
            key: value
            for key, value in parsed.items()
            if key
            not in {
                "doc_id",
                "id",
                "title",
                "name",
                "source",
                "path",
                "content",
                "text",
                "body",
                "metadata",
            }
        }
        | metadata,
    }


def normalize_import_input(payload: KnowledgeImportInput) -> KnowledgeImportInput:
    """归一化导入输入，兼容 txt/md/json 三类正文。"""

    source_type = _normalize_source_type(payload.source_type, payload.source)
    if source_type != "json":
        return KnowledgeImportInput(
            content=payload.content.strip(),
            doc_id=payload.doc_id.strip(),
            title=payload.title.strip(),
            source=payload.source.strip(),
            source_type=source_type,
            metadata=dict(payload.metadata or {}),
        )

    parsed = _parse_json_document(payload.content)
    content = str(parsed.get("content") or payload.content).strip()
    metadata = dict(parsed.get("metadata") or {})
    metadata.update(payload.metadata or {})
    return KnowledgeImportInput(
        content=content,
        doc_id=(payload.doc_id or parsed.get("doc_id") or "").strip(),
        title=(payload.title or parsed.get("title") or "").strip(),
        source=(payload.source or parsed.get("source") or "").strip(),
        source_type=source_type,
        metadata=metadata,
    )


def build_stable_doc_id(content: str, explicit_doc_id: str = "") -> str:
    """生成稳定 doc_id；未显式传入时按内容 hash 去重。"""

    if explicit_doc_id.strip():
        return explicit_doc_id.strip()
    return f"doc-{content_hash(content)[:DOC_ID_HASH_CHARS]}"


def build_chunk_records(
    *,
    doc_id: str,
    title: str,
    source: str,
    content: str,
    source_type: str = "",
) -> tuple[list[KnowledgeChunkRecord], list[dict]]:
    """构造 SQLite catalog record 和 Chroma upsert record。"""

    catalog_records: list[KnowledgeChunkRecord] = []
    chroma_records: list[dict] = []

    for chunk in chunk_document_text(
        doc_id=doc_id,
        text=content,
        source_type=source_type,
    ):
        metadata = {
            "doc_id": chunk.doc_id,
            "doc_title": title,
            "source": source,
            "section_title": chunk.section_title,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "chunk_char_len": chunk.char_len,
        }
        catalog_records.append(
            KnowledgeChunkRecord(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=title,
                source=source,
                section_title=chunk.section_title,
                chunk_index=chunk.chunk_index,
                content=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                chunk_char_len=chunk.char_len,
                metadata=metadata,
            )
        )
        chroma_records.append(
            {
                "id": chunk.chunk_id,
                "document": chunk.text,
                "metadata": metadata,
            }
        )

    return catalog_records, chroma_records


def _upsert_chroma_records(
    *,
    store: ChromaVectorStore,
    doc_id: str,
    records: list[dict],
    batch_size: int,
) -> None:
    """把 chunk 写入 Chroma；先删旧 doc_id，避免重导入残留旧 chunk。"""

    store.delete(
        collection_name=VECTOR_STORE_CONFIG.doc_collection_name,
        where={"doc_id": doc_id},
    )

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start : batch_start + batch_size]
        texts = [item["document"] for item in batch]
        embeddings = embed_texts(texts, profile="doc_embedding")
        store.upsert(
            collection_name=VECTOR_STORE_CONFIG.doc_collection_name,
            ids=[item["id"] for item in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[item["metadata"] for item in batch],
        )


def import_knowledge_document(
    payload: KnowledgeImportInput,
    *,
    catalog: KnowledgeCatalog | None = None,
    vector_store: ChromaVectorStore | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> KnowledgeImportResult:
    """导入单篇文档，并同步写入 SQLite FTS5 与 Chroma。"""

    normalized = normalize_import_input(payload)
    if not normalized.content:
        raise ValueError("content must not be empty")

    doc_id = build_stable_doc_id(normalized.content, normalized.doc_id)
    source = normalized.source or f"{doc_id}.{normalized.source_type}"
    title = _infer_title(
        title=normalized.title,
        source=source,
        content=normalized.content,
    )
    catalog_records, chroma_records = build_chunk_records(
        doc_id=doc_id,
        title=title,
        source=source,
        content=normalized.content,
        source_type=normalized.source_type,
    )

    active_catalog = catalog or KnowledgeCatalog()
    active_catalog.upsert_document(
        doc_id=doc_id,
        title=title,
        source=source,
        content=normalized.content,
        source_type=normalized.source_type,
        metadata=normalized.metadata,
    )
    active_catalog.replace_chunks(catalog_records)

    active_store = vector_store or ChromaVectorStore()
    _upsert_chroma_records(
        store=active_store,
        doc_id=doc_id,
        records=chroma_records,
        batch_size=batch_size,
    )

    return KnowledgeImportResult(
        doc_id=doc_id,
        title=title,
        source=source,
        source_type=normalized.source_type,
        content_hash=content_hash(normalized.content),
        content_char_len=len(normalized.content),
        chunk_count=len(catalog_records),
        indexed_to_sqlite=True,
        indexed_to_chroma=True,
    )

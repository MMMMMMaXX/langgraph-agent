"""知识库管理服务。

导入解决“怎么写入”，这里解决“怎么维护”：
- 删除文档时同时清 SQLite/FTS5 和 Chroma；
- Chroma 损坏或 embedding 策略变化时，可以从 SQLite catalog 重建 dense index。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import VECTOR_STORE_CONFIG
from app.knowledge.catalog import KnowledgeCatalog
from app.llm import embed_texts
from app.vector_store import ChromaVectorStore

DEFAULT_BATCH_SIZE = 16


@dataclass(frozen=True)
class KnowledgeDeleteResult:
    doc_id: str
    deleted: bool
    chunk_count: int
    deleted_from_sqlite: bool
    deleted_from_chroma: bool


@dataclass(frozen=True)
class KnowledgeReindexResult:
    doc_id: str
    doc_count: int
    chunk_count: int
    reindexed_to_chroma: bool


def _chunk_to_chroma_record(chunk: dict) -> dict:
    metadata = {
        "doc_id": chunk["doc_id"],
        "doc_title": chunk["doc_title"],
        "source": chunk["source"],
        "section_title": chunk.get("section_title", ""),
        "chunk_index": chunk["chunk_index"],
        "start_char": chunk["start_char"],
        "end_char": chunk["end_char"],
        "chunk_char_len": chunk["chunk_char_len"],
    }
    metadata.update(chunk.get("metadata") or {})
    return {
        "id": chunk["chunk_id"],
        "document": chunk["content"],
        "metadata": metadata,
    }


def _upsert_chroma_records(
    *,
    store: ChromaVectorStore,
    records: list[dict],
    batch_size: int,
) -> None:
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


def delete_knowledge_document(
    doc_id: str,
    *,
    catalog: KnowledgeCatalog | None = None,
    vector_store: ChromaVectorStore | None = None,
) -> KnowledgeDeleteResult:
    """删除单篇知识库文档，同时清理 Chroma 中对应 chunk。"""

    active_catalog = catalog or KnowledgeCatalog()
    active_store = vector_store or ChromaVectorStore()

    sqlite_result = active_catalog.delete_document(doc_id)
    deleted_from_sqlite = bool(sqlite_result["deleted"])
    deleted_from_chroma = False
    if deleted_from_sqlite:
        active_store.delete(
            collection_name=VECTOR_STORE_CONFIG.doc_collection_name,
            where={"doc_id": doc_id},
        )
        deleted_from_chroma = True

    return KnowledgeDeleteResult(
        doc_id=doc_id,
        deleted=deleted_from_sqlite,
        chunk_count=int(sqlite_result["chunk_count"]),
        deleted_from_sqlite=deleted_from_sqlite,
        deleted_from_chroma=deleted_from_chroma,
    )


def reindex_knowledge_document(
    doc_id: str,
    *,
    catalog: KnowledgeCatalog | None = None,
    vector_store: ChromaVectorStore | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> KnowledgeReindexResult:
    """从 SQLite catalog 重建单篇文档的 Chroma dense index。"""

    active_catalog = catalog or KnowledgeCatalog()
    chunks = active_catalog.list_chunks(doc_id=doc_id)
    if not chunks:
        return KnowledgeReindexResult(
            doc_id=doc_id,
            doc_count=0,
            chunk_count=0,
            reindexed_to_chroma=False,
        )

    active_store = vector_store or ChromaVectorStore()
    active_store.delete(
        collection_name=VECTOR_STORE_CONFIG.doc_collection_name,
        where={"doc_id": doc_id},
    )
    _upsert_chroma_records(
        store=active_store,
        records=[_chunk_to_chroma_record(chunk) for chunk in chunks],
        batch_size=batch_size,
    )

    return KnowledgeReindexResult(
        doc_id=doc_id,
        doc_count=1,
        chunk_count=len(chunks),
        reindexed_to_chroma=True,
    )


def reindex_all_knowledge_documents(
    *,
    catalog: KnowledgeCatalog | None = None,
    vector_store: ChromaVectorStore | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> KnowledgeReindexResult:
    """从 SQLite catalog 全量重建 Chroma docs collection。"""

    active_catalog = catalog or KnowledgeCatalog()
    chunks = active_catalog.list_chunks()
    active_store = vector_store or ChromaVectorStore()
    active_store.reset_collection(
        VECTOR_STORE_CONFIG.doc_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if chunks:
        _upsert_chroma_records(
            store=active_store,
            records=[_chunk_to_chroma_record(chunk) for chunk in chunks],
            batch_size=batch_size,
        )

    return KnowledgeReindexResult(
        doc_id="*",
        doc_count=len({chunk["doc_id"] for chunk in chunks}),
        chunk_count=len(chunks),
        reindexed_to_chroma=True,
    )

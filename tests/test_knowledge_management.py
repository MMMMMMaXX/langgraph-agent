from __future__ import annotations

from typing import Any

import app.knowledge.management as management_mod
from app.knowledge import KnowledgeCatalog, KnowledgeChunkRecord
from app.knowledge.management import (
    delete_knowledge_document,
    reindex_all_knowledge_documents,
    reindex_knowledge_document,
)


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted: list[dict[str, Any]] = []
        self.upserts: list[dict[str, Any]] = []
        self.reset_collections: list[dict[str, Any]] = []

    def delete(self, **kwargs) -> None:
        self.deleted.append(kwargs)

    def upsert(self, **kwargs) -> None:
        self.upserts.append(kwargs)

    def reset_collection(self, name: str, metadata: dict | None = None) -> None:
        self.reset_collections.append({"name": name, "metadata": metadata})


def _seed_catalog(catalog: KnowledgeCatalog, doc_id: str = "doc1") -> None:
    catalog.reset()
    catalog.upsert_document(
        doc_id=doc_id,
        title="测试文档",
        source="unit.md",
        content="Skill 是能力模块。",
        source_type="md",
    )
    catalog.replace_chunks(
        [
            KnowledgeChunkRecord(
                chunk_id=f"{doc_id}::chunk::0",
                doc_id=doc_id,
                doc_title="测试文档",
                source="unit.md",
                section_title="简介",
                chunk_index=0,
                content="Skill 是能力模块。",
                start_char=0,
                end_char=12,
                chunk_char_len=12,
            )
        ]
    )


def test_delete_knowledge_document_removes_catalog_and_chroma(tmp_path) -> None:
    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    _seed_catalog(catalog)
    fake_store = FakeVectorStore()

    result = delete_knowledge_document(
        "doc1",
        catalog=catalog,
        vector_store=fake_store,
    )

    assert result.deleted is True
    assert result.chunk_count == 1
    assert catalog.get_document("doc1") is None
    assert catalog.search_chunks("Skill", top_k=3) == []
    assert fake_store.deleted[0]["where"] == {"doc_id": "doc1"}


def test_reindex_knowledge_document_rebuilds_chroma_from_catalog(
    tmp_path,
    monkeypatch,
) -> None:
    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    _seed_catalog(catalog)
    fake_store = FakeVectorStore()
    monkeypatch.setattr(
        management_mod,
        "embed_texts",
        lambda texts, profile: [[0.1, 0.2, 0.3] for _ in texts],
    )

    result = reindex_knowledge_document(
        "doc1",
        catalog=catalog,
        vector_store=fake_store,
    )

    assert result.reindexed_to_chroma is True
    assert result.chunk_count == 1
    assert fake_store.deleted[0]["where"] == {"doc_id": "doc1"}
    assert fake_store.upserts[0]["ids"] == ["doc1::chunk::0"]
    assert fake_store.upserts[0]["metadatas"][0]["doc_title"] == "测试文档"


def test_reindex_all_knowledge_documents_resets_collection(
    tmp_path,
    monkeypatch,
) -> None:
    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    _seed_catalog(catalog)
    fake_store = FakeVectorStore()
    monkeypatch.setattr(
        management_mod,
        "embed_texts",
        lambda texts, profile: [[0.1, 0.2, 0.3] for _ in texts],
    )

    result = reindex_all_knowledge_documents(
        catalog=catalog,
        vector_store=fake_store,
    )

    assert result.doc_id == "*"
    assert result.doc_count == 1
    assert result.chunk_count == 1
    assert fake_store.reset_collections[0]["name"] == "docs"
    assert fake_store.upserts[0]["ids"] == ["doc1::chunk::0"]

from __future__ import annotations

from typing import Any

import app.knowledge.ingestion as ingestion_mod
from app.knowledge import KnowledgeCatalog, KnowledgeImportInput
from app.knowledge.ingestion import import_knowledge_document


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted: list[dict[str, Any]] = []
        self.upserts: list[dict[str, Any]] = []

    def delete(self, **kwargs) -> None:
        self.deleted.append(kwargs)

    def upsert(self, **kwargs) -> None:
        self.upserts.append(kwargs)


def test_import_knowledge_document_writes_catalog_and_vector(
    tmp_path,
    monkeypatch,
) -> None:
    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    fake_store = FakeVectorStore()

    monkeypatch.setattr(
        ingestion_mod,
        "embed_texts",
        lambda texts, profile: [[0.1, 0.2, 0.3] for _ in texts],
    )

    content = "# WAI-ARIA\n\nWAI-ARIA 是无障碍技术规范，可以帮助屏幕阅读器识别状态。"
    result = import_knowledge_document(
        KnowledgeImportInput(
            title="无障碍指南",
            source="accessibility.md",
            source_type="md",
            content=content,
        ),
        catalog=catalog,
        vector_store=fake_store,
    )

    assert result.doc_id.startswith("doc-")
    assert result.title == "无障碍指南"
    assert result.content_char_len == len(content)
    assert result.chunk_count >= 1
    assert result.indexed_to_sqlite is True
    assert result.indexed_to_chroma is True

    hits = catalog.search_chunks("WAI-ARIA 是什么", top_k=3)
    assert hits
    assert hits[0]["doc_title"] == "无障碍指南"

    assert fake_store.deleted[0]["where"] == {"doc_id": result.doc_id}
    assert fake_store.upserts
    assert fake_store.upserts[0]["collection_name"] == "docs"
    assert fake_store.upserts[0]["metadatas"][0]["doc_title"] == "无障碍指南"

    document_content = catalog.get_document_content(result.doc_id)
    assert document_content is not None
    assert document_content["content_text"] == content


def test_import_knowledge_document_parses_json_payload(tmp_path, monkeypatch) -> None:
    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    fake_store = FakeVectorStore()
    monkeypatch.setattr(
        ingestion_mod,
        "embed_texts",
        lambda texts, profile: [[0.1, 0.2, 0.3] for _ in texts],
    )

    result = import_knowledge_document(
        KnowledgeImportInput(
            source_type="json",
            content=(
                '{"id":"doc-json","title":"虚拟列表",'
                '"content":"虚拟列表是一种只渲染可见区域的数据渲染技术。",'
                '"metadata":{"topic":"frontend"}}'
            ),
        ),
        catalog=catalog,
        vector_store=fake_store,
    )

    document = catalog.get_document("doc-json")

    assert result.doc_id == "doc-json"
    assert result.content_char_len == len("虚拟列表是一种只渲染可见区域的数据渲染技术。")
    assert document is not None
    assert document["title"] == "虚拟列表"
    assert document["content_char_len"] == result.content_char_len
    assert document["metadata"]["topic"] == "frontend"

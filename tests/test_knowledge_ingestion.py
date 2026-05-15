from __future__ import annotations

from typing import Any

import app.knowledge.ingestion as ingestion_mod
from app.constants.knowledge import INGESTION_SKIPPED_REASON_CONTENT_UNCHANGED
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
    assert result.skipped_reason is None

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
    assert result.content_char_len == len(
        "虚拟列表是一种只渲染可见区域的数据渲染技术。"
    )
    assert document is not None
    assert document["title"] == "虚拟列表"
    assert document["content_char_len"] == result.content_char_len
    assert document["metadata"]["topic"] == "frontend"


def test_import_knowledge_document_skips_when_content_unchanged(
    tmp_path,
    monkeypatch,
) -> None:
    """PR-7：catalog 中已有同 doc_id 且 content_hash 一致时，跳过 chunk 重建与
    Chroma upsert，避免反复导入相同 fixture 撑爆 HNSW 段文件。"""

    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    fake_store = FakeVectorStore()
    monkeypatch.setattr(
        ingestion_mod,
        "embed_texts",
        lambda texts, profile: [[0.1, 0.2, 0.3] for _ in texts],
    )

    payload = KnowledgeImportInput(
        title="幂等测试文档",
        source="idempotent.md",
        source_type="md",
        content="# 标题\n\n这段内容用于验证 content_hash 幂等。",
    )

    first = import_knowledge_document(payload, catalog=catalog, vector_store=fake_store)
    assert first.indexed_to_sqlite is True
    assert first.indexed_to_chroma is True
    assert first.skipped_reason is None
    upserts_after_first = len(fake_store.upserts)
    deletes_after_first = len(fake_store.deleted)
    assert upserts_after_first >= 1
    assert deletes_after_first == 1

    second = import_knowledge_document(
        payload, catalog=catalog, vector_store=fake_store
    )

    assert second.doc_id == first.doc_id
    assert second.content_hash == first.content_hash
    assert second.chunk_count == first.chunk_count
    assert second.indexed_to_sqlite is False
    assert second.indexed_to_chroma is False
    assert second.skipped_reason == INGESTION_SKIPPED_REASON_CONTENT_UNCHANGED
    # 关键不变量：第二次导入既不能再写 Chroma，也不能触发先 delete 再 upsert。
    assert len(fake_store.upserts) == upserts_after_first
    assert len(fake_store.deleted) == deletes_after_first


def test_import_knowledge_document_reindexes_when_content_changes(
    tmp_path,
    monkeypatch,
) -> None:
    """PR-7：显式 doc_id 下，内容变化必须触发重新切片 + Chroma 重建。"""

    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    fake_store = FakeVectorStore()
    monkeypatch.setattr(
        ingestion_mod,
        "embed_texts",
        lambda texts, profile: [[0.1, 0.2, 0.3] for _ in texts],
    )

    first = import_knowledge_document(
        KnowledgeImportInput(
            doc_id="doc-stable",
            title="可变文档",
            source="mutable.md",
            source_type="md",
            content="# 版本 1\n\n旧内容。",
        ),
        catalog=catalog,
        vector_store=fake_store,
    )
    assert first.skipped_reason is None
    upserts_after_first = len(fake_store.upserts)
    deletes_after_first = len(fake_store.deleted)

    second = import_knowledge_document(
        KnowledgeImportInput(
            doc_id="doc-stable",
            title="可变文档",
            source="mutable.md",
            source_type="md",
            content="# 版本 2\n\n新内容更长一些，hash 一定不同。",
        ),
        catalog=catalog,
        vector_store=fake_store,
    )

    assert second.doc_id == "doc-stable"
    assert second.content_hash != first.content_hash
    assert second.indexed_to_sqlite is True
    assert second.indexed_to_chroma is True
    assert second.skipped_reason is None
    assert len(fake_store.upserts) > upserts_after_first
    assert len(fake_store.deleted) == deletes_after_first + 1


def test_import_knowledge_document_rebuilds_when_metadata_changes(
    tmp_path,
    monkeypatch,
) -> None:
    """PR-7 P1：仅 metadata（如 ACL/标签）变化时也要落库；不能因内容相同而吞掉。

    收紧的短路条件：必须 content_hash + title + source + source_type + metadata
    全部一致才跳过；任意一项变化都走完整重建路径，保证 catalog 元数据不陈旧。
    """

    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    fake_store = FakeVectorStore()
    monkeypatch.setattr(
        ingestion_mod,
        "embed_texts",
        lambda texts, profile: [[0.1, 0.2, 0.3] for _ in texts],
    )

    base_kwargs: dict[str, Any] = dict(
        doc_id="doc-acl",
        title="ACL 测试文档",
        source="acl.md",
        source_type="md",
        content="# ACL\n\n这段内容保持不变，只更新 metadata。",
    )

    first = import_knowledge_document(
        KnowledgeImportInput(**base_kwargs, metadata={"acl": ["team-a"]}),
        catalog=catalog,
        vector_store=fake_store,
    )
    assert first.skipped_reason is None
    upserts_after_first = len(fake_store.upserts)

    # metadata 变化：内容、标题、source 都不变，但 ACL 从 team-a 改成 team-b
    second = import_knowledge_document(
        KnowledgeImportInput(**base_kwargs, metadata={"acl": ["team-b"]}),
        catalog=catalog,
        vector_store=fake_store,
    )
    assert second.skipped_reason is None
    assert second.indexed_to_sqlite is True
    assert second.indexed_to_chroma is True
    document = catalog.get_document("doc-acl")
    assert document is not None
    assert document["metadata"] == {"acl": ["team-b"]}
    assert len(fake_store.upserts) > upserts_after_first

    # 第三次：完全相同输入（含 metadata），这时才应触发短路
    third = import_knowledge_document(
        KnowledgeImportInput(**base_kwargs, metadata={"acl": ["team-b"]}),
        catalog=catalog,
        vector_store=fake_store,
    )
    assert third.skipped_reason == INGESTION_SKIPPED_REASON_CONTENT_UNCHANGED
    assert third.indexed_to_sqlite is False
    assert third.indexed_to_chroma is False


def test_import_knowledge_document_rebuilds_when_title_changes(
    tmp_path,
    monkeypatch,
) -> None:
    """PR-7 P1：仅 title 变化也要走重建，避免 catalog/chunk metadata 标题陈旧。"""

    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    fake_store = FakeVectorStore()
    monkeypatch.setattr(
        ingestion_mod,
        "embed_texts",
        lambda texts, profile: [[0.1, 0.2, 0.3] for _ in texts],
    )

    content = "这是一段不会变的正文，hash 始终相同。"
    first = import_knowledge_document(
        KnowledgeImportInput(
            doc_id="doc-title",
            title="旧标题",
            source="title.txt",
            source_type="txt",
            content=content,
        ),
        catalog=catalog,
        vector_store=fake_store,
    )
    upserts_after_first = len(fake_store.upserts)

    second = import_knowledge_document(
        KnowledgeImportInput(
            doc_id="doc-title",
            title="新标题",
            source="title.txt",
            source_type="txt",
            content=content,
        ),
        catalog=catalog,
        vector_store=fake_store,
    )

    assert first.title == "旧标题"
    assert second.title == "新标题"
    assert second.skipped_reason is None
    assert second.indexed_to_sqlite is True
    assert len(fake_store.upserts) > upserts_after_first
    document = catalog.get_document("doc-title")
    assert document is not None
    assert document["title"] == "新标题"

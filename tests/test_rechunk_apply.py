import sqlite3
from typing import Any

import app.knowledge.management as management_mod
import app.knowledge.rechunk_apply as rechunk_apply_mod
from app.constants.knowledge import (
    DOCUMENT_CONTENT_CHAR_LEN_COLUMN,
    DOCUMENT_CONTENT_TEXT_COLUMN,
    RECHUNK_ERROR_DOCUMENT_CONTENT_MISSING,
    RECHUNK_SOURCE_MODE_DOCUMENT_CONTENT,
)
from app.knowledge import KnowledgeCatalog, KnowledgeChunkRecord
from app.knowledge.rechunk_apply import apply_rechunk_document
from app.knowledge.rechunk_common import RechunkPreviewParams


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted: list[dict[str, Any]] = []
        self.upserts: list[dict[str, Any]] = []

    def delete(self, **kwargs) -> None:
        self.deleted.append(kwargs)

    def upsert(self, **kwargs) -> None:
        self.upserts.append(kwargs)


def _seed_doc(catalog: KnowledgeCatalog) -> None:
    content = (
        "# Skills\n\n"
        "Agent Skills 是模块化能力单元，可以扩展助手能力。\n\n"
        "它由说明文件、脚本和资源组成。"
    )
    catalog.reset()
    catalog.upsert_document(
        doc_id="doc-skills",
        title="Skills 构建指南",
        source="skills.md",
        source_type="md",
        content=content,
    )
    catalog.replace_chunks(
        [
            KnowledgeChunkRecord(
                chunk_id="doc-skills::chunk::0",
                doc_id="doc-skills",
                doc_title="Skills 构建指南",
                source="skills.md",
                section_title="Skills",
                chunk_index=0,
                content=content,
                start_char=0,
                end_char=len(content),
                chunk_char_len=len(content),
            )
        ]
    )


def test_apply_rechunk_document_replaces_chunks_and_reindexes_chroma(
    tmp_path,
    monkeypatch,
) -> None:
    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    _seed_doc(catalog)
    fake_store = FakeVectorStore()
    monkeypatch.setattr(
        management_mod,
        "embed_texts",
        lambda texts, profile: [[0.1, 0.2, 0.3] for _ in texts],
    )

    report = apply_rechunk_document(
        "doc-skills",
        catalog=catalog,
        vector_store=fake_store,
        params=RechunkPreviewParams(
            chunk_size_chars=40,
            chunk_overlap_chars=10,
            min_chunk_chars=10,
            sample_limit=2,
        ),
    )

    chunks = catalog.list_chunks(doc_id="doc-skills")

    assert report.applied is True
    assert report.source_mode == RECHUNK_SOURCE_MODE_DOCUMENT_CONTENT
    assert report.old_chunk_count == 1
    assert report.new_chunk_count == len(chunks)
    assert report.reindexed_to_chroma is True
    assert report.preview.chunk_count == len(chunks)
    assert fake_store.deleted[0]["where"] == {"doc_id": "doc-skills"}
    assert fake_store.upserts[0]["ids"] == [chunk["chunk_id"] for chunk in chunks]


def test_apply_rechunk_document_requires_stored_original_content(tmp_path) -> None:
    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    _seed_doc(catalog)
    with sqlite3.connect(catalog.path) as conn:
        conn.execute(
            f"""
            UPDATE documents
            SET {DOCUMENT_CONTENT_TEXT_COLUMN} = '',
                {DOCUMENT_CONTENT_CHAR_LEN_COLUMN} = 0
            WHERE doc_id = ?
            """,
            ("doc-skills",),
        )

    try:
        apply_rechunk_document("doc-skills", catalog=catalog)
    except ValueError as exc:
        assert str(exc) == RECHUNK_ERROR_DOCUMENT_CONTENT_MISSING
    else:
        raise AssertionError("expected ValueError")


def test_apply_rechunk_document_rolls_back_sqlite_when_reindex_fails(
    tmp_path,
    monkeypatch,
) -> None:
    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    _seed_doc(catalog)
    original_chunks = catalog.list_chunks(doc_id="doc-skills")
    calls = {"count": 0}

    def flaky_reindex(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("embedding service unavailable")
        return type(
            "Result",
            (),
            {"reindexed_to_chroma": True},
        )()

    monkeypatch.setattr(rechunk_apply_mod, "reindex_knowledge_document", flaky_reindex)

    try:
        apply_rechunk_document(
            "doc-skills",
            catalog=catalog,
            params=RechunkPreviewParams(
                chunk_size_chars=40,
                chunk_overlap_chars=10,
                min_chunk_chars=10,
            ),
        )
    except RuntimeError as exc:
        assert str(exc) == "embedding service unavailable"
    else:
        raise AssertionError("expected RuntimeError")

    restored_chunks = catalog.list_chunks(doc_id="doc-skills")
    assert calls["count"] == 2
    assert [chunk["chunk_id"] for chunk in restored_chunks] == [
        chunk["chunk_id"] for chunk in original_chunks
    ]
    assert [chunk["content"] for chunk in restored_chunks] == [
        chunk["content"] for chunk in original_chunks
    ]

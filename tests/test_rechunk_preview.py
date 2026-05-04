import sqlite3

from app.constants.knowledge import (
    DOCUMENT_CONTENT_CHAR_LEN_COLUMN,
    DOCUMENT_CONTENT_TEXT_COLUMN,
    RECHUNK_SOURCE_MODE_DOCUMENT_CONTENT,
    RECHUNK_SOURCE_MODE_RECONSTRUCTED_FROM_CHUNKS,
    RECHUNK_WARNING_SOURCE_RECONSTRUCTED,
)
from app.knowledge import KnowledgeCatalog, KnowledgeChunkRecord
from app.knowledge.rechunk_preview import (
    RechunkPreviewParams,
    preview_rechunk_document,
)


def _seed_doc(catalog: KnowledgeCatalog) -> None:
    content = "# Skills\n\nAgent Skills 是模块化能力单元。\n\n它由说明文件和脚本组成。"
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
                content="# Skills\n\nAgent Skills 是模块化能力单元。",
                start_char=0,
                end_char=34,
                chunk_char_len=34,
            ),
            KnowledgeChunkRecord(
                chunk_id="doc-skills::chunk::1",
                doc_id="doc-skills",
                doc_title="Skills 构建指南",
                source="skills.md",
                section_title="Skills",
                chunk_index=1,
                content="它由说明文件和脚本组成。",
                start_char=35,
                end_char=48,
                chunk_char_len=13,
            ),
        ]
    )


def test_preview_rechunk_document_returns_dry_run_report(tmp_path) -> None:
    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    _seed_doc(catalog)

    report = preview_rechunk_document(
        "doc-skills",
        catalog=catalog,
        params=RechunkPreviewParams(
            chunk_size_chars=80,
            chunk_overlap_chars=10,
            min_chunk_chars=10,
            sample_limit=2,
        ),
    )

    assert report.doc_id == "doc-skills"
    assert report.applied is False
    assert report.source_mode == RECHUNK_SOURCE_MODE_DOCUMENT_CONTENT
    assert report.current.chunk_count == 2
    assert report.preview.chunk_count >= 1
    assert report.warnings == []
    assert "chunk_count" in report.delta
    assert len(report.preview.samples) <= 2
    assert len(catalog.list_chunks(doc_id="doc-skills")) == 2


def test_preview_rechunk_document_falls_back_for_legacy_rows(tmp_path) -> None:
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

    report = preview_rechunk_document(
        "doc-skills",
        catalog=catalog,
        params=RechunkPreviewParams(
            chunk_size_chars=80,
            chunk_overlap_chars=10,
            min_chunk_chars=10,
            sample_limit=2,
        ),
    )

    assert report.source_mode == RECHUNK_SOURCE_MODE_RECONSTRUCTED_FROM_CHUNKS
    assert report.warnings == [RECHUNK_WARNING_SOURCE_RECONSTRUCTED]


def test_preview_rechunk_document_validates_params(tmp_path) -> None:
    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    _seed_doc(catalog)

    try:
        preview_rechunk_document(
            "doc-skills",
            catalog=catalog,
            params=RechunkPreviewParams(
                chunk_size_chars=40,
                chunk_overlap_chars=40,
                min_chunk_chars=10,
            ),
        )
    except ValueError as exc:
        assert str(exc) == "chunk_overlap_chars must be smaller than chunk_size_chars"
    else:
        raise AssertionError("expected ValueError")


def test_preview_rechunk_document_missing_doc(tmp_path) -> None:
    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    catalog.reset()

    try:
        preview_rechunk_document("missing", catalog=catalog)
    except ValueError as exc:
        assert str(exc) == "document not found"
    else:
        raise AssertionError("expected ValueError")

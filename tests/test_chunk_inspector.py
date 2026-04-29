from app.knowledge import KnowledgeCatalog, KnowledgeChunkRecord
from app.knowledge.chunk_inspector import (
    ChunkQualityThresholds,
    build_chunk_quality_report,
    inspect_document_chunks,
)


def test_build_chunk_quality_report_marks_short_long_and_sections() -> None:
    chunks = [
        {
            "chunk_id": "doc1::chunk::0",
            "chunk_index": 0,
            "section_title": "Intro",
            "chunk_char_len": 20,
            "content": "short chunk",
        },
        {
            "chunk_id": "doc1::chunk::1",
            "chunk_index": 1,
            "section_title": "Intro",
            "chunk_char_len": 100,
            "content": "normal chunk content",
        },
        {
            "chunk_id": "doc1::chunk::2",
            "chunk_index": 2,
            "section_title": "Usage",
            "chunk_char_len": 220,
            "content": "long chunk content",
        },
    ]

    report = build_chunk_quality_report(
        doc_id="doc1",
        chunks=chunks,
        thresholds=ChunkQualityThresholds(short_chars=40, long_chars=200),
        sample_limit=2,
    )

    assert report.chunk_count == 3
    assert report.total_chars == 340
    assert report.min_chars == 20
    assert report.max_chars == 220
    assert report.avg_chars == 113.33
    assert report.median_chars == 100
    assert report.short_chunk_count == 1
    assert report.long_chunk_count == 1
    assert report.section_count == 2
    assert report.top_sections[0] == {"section_title": "Intro", "chunk_count": 2}
    assert len(report.samples) == 2
    assert "has_short_chunks" in report.warnings
    assert "has_long_chunks" in report.warnings


def test_inspect_document_chunks_reads_catalog(tmp_path) -> None:
    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    catalog.reset()
    catalog.upsert_document(
        doc_id="doc1",
        title="Skills 构建指南",
        source="skills.md",
        source_type="md",
        content="# Skills\n\nUse skills to package workflows.",
    )
    catalog.replace_chunks(
        [
            KnowledgeChunkRecord(
                chunk_id="doc1::chunk::0",
                doc_id="doc1",
                doc_title="Skills 构建指南",
                source="skills.md",
                section_title="Skills",
                chunk_index=0,
                content="Use skills to package workflows.",
                start_char=0,
                end_char=32,
                chunk_char_len=32,
            ),
            KnowledgeChunkRecord(
                chunk_id="doc1::chunk::1",
                doc_id="doc1",
                doc_title="Skills 构建指南",
                source="skills.md",
                section_title="Skills",
                chunk_index=1,
                content="A skill includes instructions and optional assets.",
                start_char=33,
                end_char=82,
                chunk_char_len=49,
            ),
        ]
    )

    report = inspect_document_chunks(
        "doc1",
        catalog=catalog,
        thresholds=ChunkQualityThresholds(short_chars=40, long_chars=100),
    )

    assert report.doc_id == "doc1"
    assert report.chunk_count == 2
    assert report.short_chunk_count == 1
    assert report.section_count == 1
    assert report.samples[0]["chunk_id"] == "doc1::chunk::0"

from __future__ import annotations

from typing import Any

from app.constants.knowledge import (
    KNOWLEDGE_HEALTH_STATUS_OK,
    KNOWLEDGE_HEALTH_STATUS_WARN,
    KNOWLEDGE_HEALTH_WARNING_MISSING_CHROMA_CHUNKS,
)
from app.knowledge import KnowledgeCatalog, KnowledgeChunkRecord
from app.knowledge.health import inspect_knowledge_health


class FakeVectorStore:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def count(self, **kwargs: Any) -> int:
        return len(self.ids)

    def get(self, **kwargs: Any) -> dict:
        return {"ids": self.ids, "documents": [], "metadatas": []}


def _seed_catalog(catalog: KnowledgeCatalog) -> None:
    catalog.reset()
    catalog.upsert_document(
        doc_id="doc-health",
        title="健康检查文档",
        source="health.md",
        content="Skill 是能力模块。",
        source_type="md",
    )
    catalog.replace_chunks(
        [
            KnowledgeChunkRecord(
                chunk_id="doc-health::chunk::0",
                doc_id="doc-health",
                doc_title="健康检查文档",
                source="health.md",
                chunk_index=0,
                content="Skill 是能力模块。",
                start_char=0,
                end_char=12,
                chunk_char_len=12,
            )
        ]
    )


def test_knowledge_health_reports_ok_when_indexes_match(tmp_path) -> None:
    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    _seed_catalog(catalog)

    report = inspect_knowledge_health(
        catalog=catalog,
        vector_store=FakeVectorStore(["doc-health::chunk::0"]),
    )

    assert report.status == KNOWLEDGE_HEALTH_STATUS_OK
    assert report.sqlite["chunk_count"] == 1
    assert report.fts["ready"] is True
    assert report.chroma["chunk_count"] == 1
    assert report.consistency["exact_checked"] is True
    assert report.warnings == []


def test_knowledge_health_warns_when_chroma_missing_chunk(tmp_path) -> None:
    catalog = KnowledgeCatalog(tmp_path / "knowledge.sqlite3")
    _seed_catalog(catalog)

    report = inspect_knowledge_health(
        catalog=catalog,
        vector_store=FakeVectorStore([]),
    )

    assert report.status == KNOWLEDGE_HEALTH_STATUS_WARN
    assert KNOWLEDGE_HEALTH_WARNING_MISSING_CHROMA_CHUNKS in report.warnings
    assert report.consistency["missing_chroma_chunk_ids"] == ["doc-health::chunk::0"]

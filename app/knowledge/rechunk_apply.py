"""知识库 rechunk apply。

preview 负责安全试算；apply 负责把候选参数真正写入 SQLite/FTS5，并同步重建
该文档的 Chroma dense index。这里刻意要求 catalog 已保存完整原文，避免把旧
chunks 拼接出的近似文本当作真原文写回索引。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.chunking import DocumentChunk, chunk_document_text
from app.constants.knowledge import (
    RECHUNK_APPLY_ROLLBACK_STAGE,
    RECHUNK_ERROR_DOCUMENT_CONTENT_MISSING,
    RECHUNK_ERROR_DOCUMENT_NOT_FOUND,
    RECHUNK_SOURCE_MODE_DOCUMENT_CONTENT,
    RECHUNK_WARNING_PREVIEW_GENERATED_NO_CHUNKS,
)
from app.knowledge.catalog import KnowledgeCatalog, KnowledgeChunkRecord
from app.knowledge.chunk_inspector import (
    ChunkQualityReport,
    ChunkQualityThresholds,
    build_chunk_quality_report,
)
from app.knowledge.management import DEFAULT_BATCH_SIZE, reindex_knowledge_document
from app.knowledge.rechunk_common import (
    RechunkPreviewParams,
    build_rechunk_delta,
    chunk_to_report_item,
    validate_rechunk_preview_params,
)
from app.utils.logger import log_warning
from app.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class RechunkApplyReport:
    """Rechunk apply 结果。

    current 是应用前的质量统计；preview 是实际写入后的候选 chunk 统计。
    命名保留 preview，是为了让调用方能直接对照 dry-run 结果。
    """

    doc_id: str
    title: str
    source: str
    source_type: str
    applied: bool
    source_mode: str
    params: dict
    current: ChunkQualityReport
    preview: ChunkQualityReport
    delta: dict
    old_chunk_count: int
    new_chunk_count: int
    reindexed_to_chroma: bool
    warnings: list[str] = field(default_factory=list)


def _chunk_to_catalog_record(
    *,
    chunk: DocumentChunk,
    document: dict,
) -> KnowledgeChunkRecord:
    """把 chunking 输出转换成 catalog 可写入的结构。"""

    return KnowledgeChunkRecord(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        doc_title=str(document.get("title", "")),
        source=str(document.get("source", "")),
        section_title=chunk.section_title,
        chunk_index=chunk.chunk_index,
        content=chunk.text,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        chunk_char_len=chunk.char_len,
        metadata={"source_type": str(document.get("source_type", ""))},
    )


def _stored_chunk_to_catalog_record(chunk: dict) -> KnowledgeChunkRecord:
    """把 catalog 读出的旧 chunk 转回可写结构，用于失败回滚。"""

    return KnowledgeChunkRecord(
        chunk_id=str(chunk.get("chunk_id", "")),
        doc_id=str(chunk.get("doc_id", "")),
        doc_title=str(chunk.get("doc_title", "")),
        source=str(chunk.get("source", "")),
        section_title=str(chunk.get("section_title", "")),
        chunk_index=int(chunk.get("chunk_index", 0)),
        content=str(chunk.get("content", "")),
        start_char=int(chunk.get("start_char", 0)),
        end_char=int(chunk.get("end_char", 0)),
        chunk_char_len=int(chunk.get("chunk_char_len", 0)),
        metadata=dict(chunk.get("metadata") or {}),
    )


def _restore_previous_chunks_after_failure(
    *,
    doc_id: str,
    catalog: KnowledgeCatalog,
    vector_store: ChromaVectorStore | None,
    batch_size: int,
    old_records: list[KnowledgeChunkRecord],
    error: Exception,
) -> None:
    """新索引重建失败时尽力恢复旧 SQLite/FTS5 和旧 Chroma。"""

    catalog.replace_chunks(old_records)
    rollback_chroma_restored = False
    try:
        rollback_result = reindex_knowledge_document(
            doc_id,
            catalog=catalog,
            vector_store=vector_store,
            batch_size=batch_size,
        )
        rollback_chroma_restored = rollback_result.reindexed_to_chroma
    except Exception as rollback_error:  # noqa: BLE001 - 回滚阶段必须保留原始异常继续抛出
        log_warning(
            RECHUNK_APPLY_ROLLBACK_STAGE,
            "failed to restore previous Chroma index after rechunk apply failure",
            {
                "doc_id": doc_id,
                "original_error": repr(error),
                "rollback_error": repr(rollback_error),
            },
        )
        return

    log_warning(
        RECHUNK_APPLY_ROLLBACK_STAGE,
        "rechunk apply failed; restored previous SQLite chunks and attempted Chroma rollback",
        {
            "doc_id": doc_id,
            "old_chunk_count": len(old_records),
            "rollback_chroma_restored": rollback_chroma_restored,
            "original_error": repr(error),
        },
    )


def apply_rechunk_document(
    doc_id: str,
    *,
    params: RechunkPreviewParams | None = None,
    catalog: KnowledgeCatalog | None = None,
    vector_store: ChromaVectorStore | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> RechunkApplyReport:
    """应用新的 chunk 参数，并重建单篇文档的 dense index。"""

    active_params = params or RechunkPreviewParams()
    validate_rechunk_preview_params(active_params)

    active_catalog = catalog or KnowledgeCatalog()
    document = active_catalog.get_document(doc_id)
    if document is None:
        raise ValueError(RECHUNK_ERROR_DOCUMENT_NOT_FOUND)

    document_content = active_catalog.get_document_content(doc_id) or {}
    source_text = str(document_content.get("content_text") or "")
    if not source_text.strip():
        raise ValueError(RECHUNK_ERROR_DOCUMENT_CONTENT_MISSING)

    current_chunks = active_catalog.list_chunks(doc_id=doc_id)
    thresholds = ChunkQualityThresholds(
        short_chars=active_params.min_chunk_chars,
        long_chars=active_params.chunk_size_chars * 2,
    )
    current_report = build_chunk_quality_report(
        doc_id=doc_id,
        chunks=current_chunks,
        thresholds=thresholds,
        sample_limit=active_params.sample_limit,
    )

    new_chunks = chunk_document_text(
        doc_id=doc_id,
        text=source_text,
        chunk_size_chars=active_params.chunk_size_chars,
        chunk_overlap_chars=active_params.chunk_overlap_chars,
        min_chunk_chars=active_params.min_chunk_chars,
        source_type=str(document.get("source_type", "")),
    )
    records = [
        _chunk_to_catalog_record(chunk=chunk, document=document)
        for chunk in new_chunks
    ]
    preview_report = build_chunk_quality_report(
        doc_id=doc_id,
        chunks=[chunk_to_report_item(chunk) for chunk in new_chunks],
        thresholds=thresholds,
        sample_limit=active_params.sample_limit,
    )

    warnings: list[str] = []
    if preview_report.chunk_count == 0:
        warnings.append(RECHUNK_WARNING_PREVIEW_GENERATED_NO_CHUNKS)

    old_records = [
        _stored_chunk_to_catalog_record(chunk)
        for chunk in current_chunks
    ]
    active_catalog.replace_chunks(records)
    try:
        reindex_result = reindex_knowledge_document(
            doc_id,
            catalog=active_catalog,
            vector_store=vector_store,
            batch_size=batch_size,
        )
    except Exception as exc:
        _restore_previous_chunks_after_failure(
            doc_id=doc_id,
            catalog=active_catalog,
            vector_store=vector_store,
            batch_size=batch_size,
            old_records=old_records,
            error=exc,
        )
        raise

    return RechunkApplyReport(
        doc_id=doc_id,
        title=str(document.get("title", "")),
        source=str(document.get("source", "")),
        source_type=str(document.get("source_type", "")),
        applied=True,
        source_mode=RECHUNK_SOURCE_MODE_DOCUMENT_CONTENT,
        params={
            "chunk_size_chars": active_params.chunk_size_chars,
            "chunk_overlap_chars": active_params.chunk_overlap_chars,
            "min_chunk_chars": active_params.min_chunk_chars,
            "sample_limit": active_params.sample_limit,
        },
        current=current_report,
        preview=preview_report,
        delta=build_rechunk_delta(current=current_report, preview=preview_report),
        old_chunk_count=len(current_chunks),
        new_chunk_count=len(records),
        reindexed_to_chroma=reindex_result.reindexed_to_chroma,
        warnings=warnings,
    )

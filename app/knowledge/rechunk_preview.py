"""知识库 rechunk dry-run。

第一版只做预览，不写 SQLite、不写 Chroma。这样可以安全比较当前 chunk 与候选
chunk 参数的差异，为后续真正的 rechunk/reindex API 提供决策依据。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.chunking import chunk_document_text
from app.constants.knowledge import (
    RECHUNK_ERROR_DOCUMENT_NOT_FOUND,
    RECHUNK_SOURCE_MODE_DOCUMENT_CONTENT,
    RECHUNK_SOURCE_MODE_RECONSTRUCTED_FROM_CHUNKS,
    RECHUNK_WARNING_PREVIEW_GENERATED_NO_CHUNKS,
    RECHUNK_WARNING_SOURCE_RECONSTRUCTED,
)
from app.knowledge.catalog import KnowledgeCatalog
from app.knowledge.chunk_inspector import (
    ChunkQualityReport,
    ChunkQualityThresholds,
    build_chunk_quality_report,
)
from app.knowledge.rechunk_common import (
    RechunkPreviewParams,
    build_rechunk_delta,
    chunk_to_report_item,
    validate_rechunk_preview_params,
)


@dataclass(frozen=True)
class RechunkPreviewReport:
    """Rechunk dry-run 报告。"""

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
    warnings: list[str] = field(default_factory=list)


def _reconstruct_text_from_chunks(chunks: list[dict]) -> str:
    """从现有 chunks 近似重建文档文本。

    这个 fallback 只服务于旧 catalog 数据：老文档没有保存完整原文，只能按
    chunk 顺序拼接内容。report.source_mode 会明确标记，避免调用方把它误认为
    严格原文。
    """

    ordered_chunks = sorted(chunks, key=lambda item: int(item.get("chunk_index", 0)))
    return "\n\n".join(
        str(chunk.get("content", "")).strip()
        for chunk in ordered_chunks
        if str(chunk.get("content", "")).strip()
    )


def preview_rechunk_document(
    doc_id: str,
    *,
    params: RechunkPreviewParams | None = None,
    catalog: KnowledgeCatalog | None = None,
) -> RechunkPreviewReport:
    """预览单篇文档在候选参数下的重新切片结果。"""

    active_params = params or RechunkPreviewParams()
    validate_rechunk_preview_params(active_params)

    active_catalog = catalog or KnowledgeCatalog()
    document = active_catalog.get_document(doc_id)
    if document is None:
        raise ValueError(RECHUNK_ERROR_DOCUMENT_NOT_FOUND)

    current_chunks = active_catalog.list_chunks(doc_id=doc_id)
    document_content = active_catalog.get_document_content(doc_id) or {}
    stored_source_text = str(document_content.get("content_text") or "")
    if stored_source_text.strip():
        source_text = stored_source_text
        source_mode = RECHUNK_SOURCE_MODE_DOCUMENT_CONTENT
        warnings: list[str] = []
    else:
        source_text = _reconstruct_text_from_chunks(current_chunks)
        source_mode = RECHUNK_SOURCE_MODE_RECONSTRUCTED_FROM_CHUNKS
        warnings = [RECHUNK_WARNING_SOURCE_RECONSTRUCTED]

    if not source_text.strip():
        raise ValueError("document has no chunks to preview")

    preview_chunks = chunk_document_text(
        doc_id=doc_id,
        text=source_text,
        chunk_size_chars=active_params.chunk_size_chars,
        chunk_overlap_chars=active_params.chunk_overlap_chars,
        min_chunk_chars=active_params.min_chunk_chars,
        source_type=str(document.get("source_type", "")),
    )
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
    preview_report = build_chunk_quality_report(
        doc_id=doc_id,
        chunks=[chunk_to_report_item(chunk) for chunk in preview_chunks],
        thresholds=thresholds,
        sample_limit=active_params.sample_limit,
    )

    if preview_report.chunk_count == 0:
        warnings.append(RECHUNK_WARNING_PREVIEW_GENERATED_NO_CHUNKS)

    return RechunkPreviewReport(
        doc_id=doc_id,
        title=str(document.get("title", "")),
        source=str(document.get("source", "")),
        source_type=str(document.get("source_type", "")),
        applied=False,
        source_mode=source_mode,
        params={
            "chunk_size_chars": active_params.chunk_size_chars,
            "chunk_overlap_chars": active_params.chunk_overlap_chars,
            "min_chunk_chars": active_params.min_chunk_chars,
            "sample_limit": active_params.sample_limit,
        },
        current=current_report,
        preview=preview_report,
        delta=build_rechunk_delta(current=current_report, preview=preview_report),
        warnings=warnings,
    )

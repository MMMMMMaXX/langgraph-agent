"""知识库 rechunk dry-run。

第一版只做预览，不写 SQLite、不写 Chroma。这样可以安全比较当前 chunk 与候选
chunk 参数的差异，为后续真正的 rechunk/reindex API 提供决策依据。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.chunking import DocumentChunk, chunk_document_text
from app.config import CHUNKING_CONFIG
from app.constants.knowledge import (
    RECHUNK_PREVIEW_DEFAULT_SAMPLE_LIMIT,
    RECHUNK_PREVIEW_MAX_CHUNK_SIZE_CHARS,
    RECHUNK_PREVIEW_MAX_MIN_CHUNK_CHARS,
    RECHUNK_PREVIEW_MAX_OVERLAP_CHARS,
    RECHUNK_PREVIEW_MIN_CHUNK_SIZE_CHARS,
    RECHUNK_PREVIEW_MIN_MIN_CHUNK_CHARS,
    RECHUNK_PREVIEW_MIN_OVERLAP_CHARS,
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


@dataclass(frozen=True)
class RechunkPreviewParams:
    """Rechunk preview 候选参数。"""

    chunk_size_chars: int = CHUNKING_CONFIG.chunk_size_chars
    chunk_overlap_chars: int = CHUNKING_CONFIG.chunk_overlap_chars
    min_chunk_chars: int = CHUNKING_CONFIG.min_chunk_chars
    sample_limit: int = RECHUNK_PREVIEW_DEFAULT_SAMPLE_LIMIT


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


def _validate_range(*, name: str, value: int, minimum: int, maximum: int) -> None:
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def validate_rechunk_preview_params(params: RechunkPreviewParams) -> None:
    """校验 dry-run 参数，避免生成无意义或过大的预览。"""

    _validate_range(
        name="chunk_size_chars",
        value=params.chunk_size_chars,
        minimum=RECHUNK_PREVIEW_MIN_CHUNK_SIZE_CHARS,
        maximum=RECHUNK_PREVIEW_MAX_CHUNK_SIZE_CHARS,
    )
    _validate_range(
        name="chunk_overlap_chars",
        value=params.chunk_overlap_chars,
        minimum=RECHUNK_PREVIEW_MIN_OVERLAP_CHARS,
        maximum=RECHUNK_PREVIEW_MAX_OVERLAP_CHARS,
    )
    _validate_range(
        name="min_chunk_chars",
        value=params.min_chunk_chars,
        minimum=RECHUNK_PREVIEW_MIN_MIN_CHUNK_CHARS,
        maximum=RECHUNK_PREVIEW_MAX_MIN_CHUNK_CHARS,
    )
    if params.chunk_overlap_chars >= params.chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")
    if params.min_chunk_chars > params.chunk_size_chars:
        raise ValueError("min_chunk_chars must be smaller than or equal to chunk_size_chars")
    if params.sample_limit < 0:
        raise ValueError("sample_limit must be greater than or equal to 0")


def _chunk_to_report_item(chunk: DocumentChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "section_title": chunk.section_title,
        "chunk_char_len": chunk.char_len,
        "content": chunk.text,
    }


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


def _build_delta(
    *,
    current: ChunkQualityReport,
    preview: ChunkQualityReport,
) -> dict:
    return {
        "chunk_count": preview.chunk_count - current.chunk_count,
        "total_chars": preview.total_chars - current.total_chars,
        "avg_chars": round(preview.avg_chars - current.avg_chars, 2),
        "median_chars": round(preview.median_chars - current.median_chars, 2),
        "short_chunk_count": preview.short_chunk_count - current.short_chunk_count,
        "long_chunk_count": preview.long_chunk_count - current.long_chunk_count,
        "section_count": preview.section_count - current.section_count,
    }


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
        raise ValueError("document not found")

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
        chunks=[_chunk_to_report_item(chunk) for chunk in preview_chunks],
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
        delta=_build_delta(current=current_report, preview=preview_report),
        warnings=warnings,
    )

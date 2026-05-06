"""Rechunk preview/apply 共用结构。

preview 和 apply 应该使用同一套参数校验与 delta 计算，否则很容易出现“预览
能过、应用失败”或“预览指标和应用指标不一致”的隐性分叉。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.chunking import DocumentChunk
from app.config import CHUNKING_CONFIG
from app.constants.knowledge import (
    RECHUNK_PREVIEW_DEFAULT_SAMPLE_LIMIT,
    RECHUNK_PREVIEW_MAX_CHUNK_SIZE_CHARS,
    RECHUNK_PREVIEW_MAX_MIN_CHUNK_CHARS,
    RECHUNK_PREVIEW_MAX_OVERLAP_CHARS,
    RECHUNK_PREVIEW_MIN_CHUNK_SIZE_CHARS,
    RECHUNK_PREVIEW_MIN_MIN_CHUNK_CHARS,
    RECHUNK_PREVIEW_MIN_OVERLAP_CHARS,
)
from app.knowledge.chunk_inspector import ChunkQualityReport


@dataclass(frozen=True)
class RechunkPreviewParams:
    """Rechunk preview/apply 候选参数。"""

    chunk_size_chars: int = CHUNKING_CONFIG.chunk_size_chars
    chunk_overlap_chars: int = CHUNKING_CONFIG.chunk_overlap_chars
    min_chunk_chars: int = CHUNKING_CONFIG.min_chunk_chars
    sample_limit: int = RECHUNK_PREVIEW_DEFAULT_SAMPLE_LIMIT


def _validate_range(*, name: str, value: int, minimum: int, maximum: int) -> None:
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def validate_rechunk_preview_params(params: RechunkPreviewParams) -> None:
    """校验 dry-run/apply 参数，避免生成无意义或过大的 chunk 集。"""

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


def chunk_to_report_item(chunk: DocumentChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "section_title": chunk.section_title,
        "chunk_char_len": chunk.char_len,
        "content": chunk.text,
    }


def build_rechunk_delta(
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

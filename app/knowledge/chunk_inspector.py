"""知识库 chunk 质量统计。

这层只做分析，不改数据。先把真实导入文档的切片质量量化出来，
再决定是否调整 chunk 参数或新增 rechunk API，避免凭感觉改检索基础设施。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, median

from app.config import CHUNKING_CONFIG
from app.knowledge.catalog import KnowledgeCatalog

DEFAULT_SAMPLE_LIMIT = 5


@dataclass(frozen=True)
class ChunkQualityThresholds:
    """chunk 质量阈值，默认跟当前 chunking config 对齐。"""

    short_chars: int = CHUNKING_CONFIG.min_chunk_chars
    long_chars: int = CHUNKING_CONFIG.chunk_size_chars * 2


@dataclass(frozen=True)
class ChunkQualityReport:
    """单篇文档的 chunk 质量报告。"""

    doc_id: str
    chunk_count: int
    total_chars: int
    min_chars: int
    max_chars: int
    avg_chars: float
    median_chars: float
    short_chunk_count: int
    long_chunk_count: int
    section_count: int
    top_sections: list[dict]
    samples: list[dict]
    warnings: list[str] = field(default_factory=list)


def _build_warnings(
    *,
    chunk_count: int,
    short_chunk_count: int,
    long_chunk_count: int,
    section_count: int,
) -> list[str]:
    warnings = []
    if chunk_count == 0:
        warnings.append("no_chunks")
    if short_chunk_count:
        warnings.append("has_short_chunks")
    if long_chunk_count:
        warnings.append("has_long_chunks")
    if chunk_count > 0 and section_count <= 1:
        warnings.append("low_section_diversity")
    return warnings


def build_chunk_quality_report(
    *,
    doc_id: str,
    chunks: list[dict],
    thresholds: ChunkQualityThresholds | None = None,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> ChunkQualityReport:
    """从 chunk 列表计算质量统计。"""

    active_thresholds = thresholds or ChunkQualityThresholds()
    lengths = [int(chunk.get("chunk_char_len", 0) or 0) for chunk in chunks]
    section_counter = Counter(
        str(chunk.get("section_title") or "(no section)") for chunk in chunks
    )
    short_count = sum(1 for length in lengths if length < active_thresholds.short_chars)
    long_count = sum(1 for length in lengths if length > active_thresholds.long_chars)
    chunk_count = len(chunks)

    samples = [
        {
            "chunk_id": chunk.get("chunk_id", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "section_title": chunk.get("section_title", ""),
            "chunk_char_len": chunk.get("chunk_char_len", 0),
            "preview": " ".join(str(chunk.get("content", "")).split())[:160],
        }
        for chunk in chunks[: max(sample_limit, 0)]
    ]

    return ChunkQualityReport(
        doc_id=doc_id,
        chunk_count=chunk_count,
        total_chars=sum(lengths),
        min_chars=min(lengths) if lengths else 0,
        max_chars=max(lengths) if lengths else 0,
        avg_chars=round(mean(lengths), 2) if lengths else 0.0,
        median_chars=round(median(lengths), 2) if lengths else 0.0,
        short_chunk_count=short_count,
        long_chunk_count=long_count,
        section_count=len(section_counter),
        top_sections=[
            {"section_title": title, "chunk_count": count}
            for title, count in section_counter.most_common(10)
        ],
        samples=samples,
        warnings=_build_warnings(
            chunk_count=chunk_count,
            short_chunk_count=short_count,
            long_chunk_count=long_count,
            section_count=len(section_counter),
        ),
    )


def inspect_document_chunks(
    doc_id: str,
    *,
    catalog: KnowledgeCatalog | None = None,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    thresholds: ChunkQualityThresholds | None = None,
) -> ChunkQualityReport:
    """读取 catalog 并生成单篇文档的 chunk 质量报告。"""

    active_catalog = catalog or KnowledgeCatalog()
    chunks = active_catalog.list_chunks(doc_id=doc_id)
    return build_chunk_quality_report(
        doc_id=doc_id,
        chunks=chunks,
        thresholds=thresholds,
        sample_limit=sample_limit,
    )

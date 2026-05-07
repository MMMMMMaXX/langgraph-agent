"""知识库健康检查服务。

这个模块只做只读检查：SQLite catalog 是知识库真相源，FTS5 和 Chroma 都是
可重建索引。health 的目标不是自动修复，而是告诉我们“是否需要 reindex”。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import VECTOR_STORE_CONFIG
from app.constants.knowledge import (
    KNOWLEDGE_HEALTH_MAX_EXACT_CHROMA_CHECK,
    KNOWLEDGE_HEALTH_STATUS_ERROR,
    KNOWLEDGE_HEALTH_STATUS_OK,
    KNOWLEDGE_HEALTH_STATUS_WARN,
    KNOWLEDGE_HEALTH_WARNING_CHROMA_COUNT_MISMATCH,
    KNOWLEDGE_HEALTH_WARNING_EXACT_CHECK_SKIPPED,
    KNOWLEDGE_HEALTH_WARNING_FTS_COUNT_MISMATCH,
    KNOWLEDGE_HEALTH_WARNING_MISSING_CHROMA_CHUNKS,
    KNOWLEDGE_HEALTH_WARNING_ORPHAN_CHROMA_CHUNKS,
)
from app.knowledge.catalog import KnowledgeCatalog
from app.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class KnowledgeHealthReport:
    """知识库索引健康检查报告。"""

    status: str
    sqlite: dict
    fts: dict
    chroma: dict
    consistency: dict
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _append_warning(warnings: list[str], code: str) -> None:
    if code not in warnings:
        warnings.append(code)


def inspect_knowledge_health(
    *,
    catalog: KnowledgeCatalog | None = None,
    vector_store: ChromaVectorStore | None = None,
    max_exact_chroma_check: int = KNOWLEDGE_HEALTH_MAX_EXACT_CHROMA_CHECK,
) -> KnowledgeHealthReport:
    """检查 SQLite / FTS5 / Chroma 的基础一致性。

    精确 id 对比只在 Chroma 记录数不超过上限时执行；大库场景先用数量检查，
    避免 health endpoint 变成昂贵的全量扫描。
    """

    active_catalog = catalog or KnowledgeCatalog()
    active_store = vector_store or ChromaVectorStore()
    warnings: list[str] = []
    errors: list[str] = []

    try:
        sqlite_stats = active_catalog.get_index_stats()
        sqlite_chunk_ids = {
            str(chunk["chunk_id"]) for chunk in active_catalog.list_chunks()
        }
    except Exception as exc:  # pragma: no cover - 真实 I/O 保护，单测覆盖正常分支
        return KnowledgeHealthReport(
            status=KNOWLEDGE_HEALTH_STATUS_ERROR,
            sqlite={"ready": False},
            fts={"ready": False},
            chroma={"ready": False},
            consistency={"exact_checked": False},
            errors=[f"sqlite_check_failed: {exc}"],
        )

    if sqlite_stats["fts_chunk_count"] != sqlite_stats["chunk_count"]:
        _append_warning(warnings, KNOWLEDGE_HEALTH_WARNING_FTS_COUNT_MISMATCH)

    chroma_ready = True
    chroma_count = 0
    chroma_ids: set[str] = set()
    exact_checked = False
    try:
        chroma_count = active_store.count(
            collection_name=VECTOR_STORE_CONFIG.doc_collection_name,
        )
        if chroma_count == 0:
            exact_checked = True
        elif chroma_count <= max(max_exact_chroma_check, 0):
            raw = active_store.get(
                collection_name=VECTOR_STORE_CONFIG.doc_collection_name,
                limit=chroma_count,
            )
            chroma_ids = {str(item_id) for item_id in raw.get("ids", [])}
            exact_checked = True
        else:
            _append_warning(warnings, KNOWLEDGE_HEALTH_WARNING_EXACT_CHECK_SKIPPED)
    except Exception as exc:  # pragma: no cover - 真实 Chroma I/O 保护
        chroma_ready = False
        errors.append(f"chroma_check_failed: {exc}")

    if chroma_ready and chroma_count != sqlite_stats["chunk_count"]:
        _append_warning(warnings, KNOWLEDGE_HEALTH_WARNING_CHROMA_COUNT_MISMATCH)

    orphan_chroma_ids: list[str] = []
    missing_chroma_ids: list[str] = []
    if exact_checked:
        orphan_chroma_ids = sorted(chroma_ids - sqlite_chunk_ids)
        missing_chroma_ids = sorted(sqlite_chunk_ids - chroma_ids)
        if orphan_chroma_ids:
            _append_warning(warnings, KNOWLEDGE_HEALTH_WARNING_ORPHAN_CHROMA_CHUNKS)
        if missing_chroma_ids:
            _append_warning(warnings, KNOWLEDGE_HEALTH_WARNING_MISSING_CHROMA_CHUNKS)

    status = KNOWLEDGE_HEALTH_STATUS_OK
    if errors:
        status = KNOWLEDGE_HEALTH_STATUS_ERROR
    elif warnings:
        status = KNOWLEDGE_HEALTH_STATUS_WARN

    return KnowledgeHealthReport(
        status=status,
        sqlite={
            "ready": True,
            "path": sqlite_stats["sqlite_path"],
            "document_count": sqlite_stats["document_count"],
            "chunk_count": sqlite_stats["chunk_count"],
        },
        fts={
            "ready": sqlite_stats["fts_ready"],
            "chunk_count": sqlite_stats["fts_chunk_count"],
        },
        chroma={
            "ready": chroma_ready,
            "collection": VECTOR_STORE_CONFIG.doc_collection_name,
            "chunk_count": chroma_count,
        },
        consistency={
            "exact_checked": exact_checked,
            "orphan_chroma_chunk_ids": orphan_chroma_ids[:20],
            "missing_chroma_chunk_ids": missing_chroma_ids[:20],
            "orphan_chroma_chunk_count": len(orphan_chroma_ids),
            "missing_chroma_chunk_count": len(missing_chroma_ids),
        },
        warnings=warnings,
        errors=errors,
    )

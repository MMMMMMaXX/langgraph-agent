from app.knowledge.catalog import KnowledgeCatalog, KnowledgeChunkRecord
from app.knowledge.chunk_inspector import (
    ChunkQualityReport,
    ChunkQualityThresholds,
    inspect_document_chunks,
)
from app.knowledge.ingestion import (
    KnowledgeImportInput,
    KnowledgeImportResult,
    import_knowledge_document,
)
from app.knowledge.management import (
    KnowledgeDeleteResult,
    KnowledgeReindexResult,
    delete_knowledge_document,
    reindex_all_knowledge_documents,
    reindex_knowledge_document,
)


def __getattr__(name: str):
    if name in {"SearchInspectReport", "inspect_retrieval"}:
        from app.knowledge.search_inspector import (
            SearchInspectReport,
            inspect_retrieval,
        )

        return {
            "SearchInspectReport": SearchInspectReport,
            "inspect_retrieval": inspect_retrieval,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "KnowledgeCatalog",
    "KnowledgeChunkRecord",
    "ChunkQualityReport",
    "ChunkQualityThresholds",
    "KnowledgeImportInput",
    "KnowledgeImportResult",
    "KnowledgeDeleteResult",
    "KnowledgeReindexResult",
    "SearchInspectReport",
    "inspect_document_chunks",
    "inspect_retrieval",
    "import_knowledge_document",
    "delete_knowledge_document",
    "reindex_knowledge_document",
    "reindex_all_knowledge_documents",
]

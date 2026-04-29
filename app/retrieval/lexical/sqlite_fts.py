from __future__ import annotations

from app.knowledge.catalog import KnowledgeCatalog


class SQLiteFtsLexicalRetriever:
    """基于 SQLite FTS5 + BM25 的 lexical retriever。"""

    def __init__(self, catalog: KnowledgeCatalog | None = None) -> None:
        self.catalog = catalog or KnowledgeCatalog()

    def search(self, query: str, top_k: int) -> list[dict]:
        return self.catalog.search_chunks(query, top_k=top_k)

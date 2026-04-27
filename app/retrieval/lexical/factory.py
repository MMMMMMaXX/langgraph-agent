from __future__ import annotations

from app.config import LEXICAL_RETRIEVAL_CONFIG
from app.retrieval.lexical.sqlite_fts import SQLiteFtsLexicalRetriever
from app.retrieval.lexical.types import LexicalRetriever

LEXICAL_BACKEND_SQLITE_FTS = "sqlite_fts"


def get_lexical_retriever() -> LexicalRetriever:
    """按配置返回 lexical retriever。

    当前只实现 SQLite FTS5。保留 factory 是为了后续接 OpenSearch 时，RAG 主链路
    不需要知道具体搜索引擎。
    """

    backend = LEXICAL_RETRIEVAL_CONFIG.backend
    if backend == LEXICAL_BACKEND_SQLITE_FTS:
        return SQLiteFtsLexicalRetriever()
    raise ValueError(f"unsupported lexical retriever backend: {backend}")

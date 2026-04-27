from __future__ import annotations

from typing import Protocol


class LexicalRetriever(Protocol):
    """Lexical 检索器协议。

    SQLite FTS5、OpenSearch、Elasticsearch 以后都实现这个接口，RAG 主链路只
    关心返回的 hit 列表，不关心底层检索引擎。
    """

    def search(self, query: str, top_k: int) -> list[dict]:
        ...

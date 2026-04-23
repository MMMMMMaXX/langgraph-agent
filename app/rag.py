"""RAG 文档检索兼容入口。

历史代码通过 `app.rag.search_docs()` 访问文档检索能力。为了避免一次性修改
所有调用点，这里保留外观层，把真实实现收敛到 `app.retrieval.doc_retrieval`。
后续新增 BM25、多路召回、相邻 chunk 合并时，优先改 retrieval 模块。
"""

from app.constants.tags import (
    CONCEPT_TAG_WEIGHTS,
    CITY_TAG_WEIGHTS,
    LITERAL_MATCH_WEIGHT,
    TAG_MATCH_TERMS,
    TAG_WEIGHTS,
    TOPIC_TAG_WEIGHTS,
)
from app.retrieval.doc_retrieval import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_HYBRID_BETA,
    RETRIEVAL_SOURCE_DENSE,
    RETRIEVAL_SOURCE_KEYWORD,
    apply_keyword_scores,
    build_doc_hit,
    dense_retrieve_docs,
    distance_to_semantic_score,
    flatten_chroma_get_result,
    flatten_chroma_query_result,
    keyword_score,
    keyword_retrieve_docs,
    merge_doc_hits,
    normalize_keyword_scores,
    normalize_keyword_text,
    rank_hybrid,
    search_docs,
)

__all__ = [
    "CITY_TAG_WEIGHTS",
    "TOPIC_TAG_WEIGHTS",
    "CONCEPT_TAG_WEIGHTS",
    "TAG_WEIGHTS",
    "TAG_MATCH_TERMS",
    "LITERAL_MATCH_WEIGHT",
    "DEFAULT_HYBRID_ALPHA",
    "DEFAULT_HYBRID_BETA",
    "RETRIEVAL_SOURCE_DENSE",
    "RETRIEVAL_SOURCE_KEYWORD",
    "normalize_keyword_text",
    "keyword_score",
    "normalize_keyword_scores",
    "distance_to_semantic_score",
    "build_doc_hit",
    "flatten_chroma_query_result",
    "flatten_chroma_get_result",
    "dense_retrieve_docs",
    "keyword_retrieve_docs",
    "merge_doc_hits",
    "apply_keyword_scores",
    "rank_hybrid",
    "search_docs",
]

"""RAG 文档检索管线。

该模块负责把“查询 -> Chroma/hybrid 检索 -> 阈值过滤 -> rerank -> chunk 合并”
封装成一个稳定结果对象。底层向量检索仍由 app.rag/search_docs 提供。
"""

from app.agents.rag.chunk_merge import merge_adjacent_doc_hits
from app.agents.rag.doc_policy import should_skip_doc_rerank
from app.agents.rag.types import DocRetrievalResult
from app.config import RAG_CONFIG, VECTOR_STORE_CONFIG
from app.rag import search_docs
from app.retrieval.reranker import rerank
from app.utils.errors import build_error_info
from app.utils.logger import now_ms


def retrieve_docs_for_rag(query: str) -> DocRetrievalResult:
    """执行 RAG 文档检索主流程。"""

    timings_ms: dict[str, float] = {}
    errors: list[str] = []

    doc_search_started_at_ms = now_ms()
    try:
        docs = search_docs(query, top_k=RAG_CONFIG.doc_top_k)
    except Exception as exc:
        docs = []
        errors.append(
            build_error_info(
                exc,
                stage="search_docs",
                source="retrieval",
                preferred_code="retrieval_error",
            )
        )
    timings_ms["docSearch"] = round(now_ms() - doc_search_started_at_ms, 2)

    filtered_docs = [doc for doc in docs if doc["score"] >= RAG_CONFIG.doc_score_threshold]

    # 对通用概念问答，语义检索往往能召回到正确文档，但最终分数未必超过固定阈值。
    # 如果 top1 结果已经有中等以上语义相关性，就允许它进入后续回答流程。
    if (
        not filtered_docs
        and docs
        and docs[0]["score"] >= RAG_CONFIG.doc_soft_match_threshold
    ):
        filtered_docs = docs[:1]

    doc_rerank_skipped, doc_rerank_skip_reason = should_skip_doc_rerank(filtered_docs)
    if doc_rerank_skipped:
        doc_hits = filtered_docs[: RAG_CONFIG.doc_rerank_top_k]
        timings_ms["docRerank"] = 0.0
    else:
        doc_rerank_started_at_ms = now_ms()
        try:
            doc_hits = rerank(
                query,
                filtered_docs,
                top_k=RAG_CONFIG.doc_rerank_top_k,
            )
        except Exception as exc:
            doc_hits = filtered_docs[: RAG_CONFIG.doc_rerank_top_k]
            errors.append(
                build_error_info(
                    exc,
                    stage="doc_rerank",
                    source="retrieval",
                    preferred_code="retrieval_error",
                )
            )
        timings_ms["docRerank"] = round(now_ms() - doc_rerank_started_at_ms, 2)

    merged_doc_hits = merge_adjacent_doc_hits(doc_hits)
    retrieval_debug = {
        "collection": VECTOR_STORE_CONFIG.doc_collection_name,
        "where": None,
        "requested_top_k": RAG_CONFIG.doc_top_k,
        "candidate_top_k": max(RAG_CONFIG.doc_top_k * 4, RAG_CONFIG.doc_top_k),
        "returned_count": len(docs),
        "filtered_count": len(filtered_docs),
        "consumed_count": len(doc_hits),
        "rerank_skipped": doc_rerank_skipped,
        "rerank_skip_reason": doc_rerank_skip_reason,
        "merged_count": len(merged_doc_hits),
        "merged": len(merged_doc_hits) != len(doc_hits),
    }

    return DocRetrievalResult(
        docs=docs,
        filtered_docs=filtered_docs,
        doc_hits=doc_hits,
        merged_doc_hits=merged_doc_hits,
        retrieval_debug=retrieval_debug,
        errors=errors,
        timings_ms=timings_ms,
    )


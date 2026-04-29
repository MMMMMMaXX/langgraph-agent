"""rag_agent 调试输出构造。

该模块只负责把内部检索命中转换成 API debug/log 友好的结构，
不参与检索和排序决策，方便后续保持调试字段稳定。
"""

from app.agents.rag.constants import (
    DOC_HIT_DEBUG_FIELDS,
    DOC_HIT_DEBUG_TEXT_PREVIEW_CHARS,
    DOC_HIT_SCORE_FIELDS,
    HYBRID_DOC_TEXT_PREVIEW_CHARS,
)
from app.constants.model_profiles import (
    PROFILE_DEFAULT_CHAT,
    PROFILE_REWRITE,
)
from app.agents.rag.types import QueryClassification, RewriteResult
from app.llm import get_profile_runtime_info
from app.utils.logger import preview, preview_hits


def preview_doc_hits(hits: list[dict]) -> list[dict]:
    """生成带 chunk 元信息的精简文档命中结果，供 debug/log 使用。

    文档检索现在读取的是 Chroma 中的 chunk 记录，只看分数不足以判断
    chunking 和索引写入是否正确。这里在保持输出可读的同时暴露稳定 id、
    原文偏移和 chunk 长度，方便直接通过 API debug 响应检查 Chroma 结构。
    """

    result = []

    for hit in hits:
        item = {
            "preview": preview(
                hit.get("content", ""),
                DOC_HIT_DEBUG_TEXT_PREVIEW_CHARS,
            ),
        }

        for field in DOC_HIT_DEBUG_FIELDS:
            if field not in hit:
                continue

            value = hit[field]
            if field in DOC_HIT_SCORE_FIELDS:
                try:
                    value = round(float(value), 4)
                except (TypeError, ValueError):
                    pass

            item[field] = value

        result.append(item)

    return result


def build_hybrid_doc_debug(hits: list[dict]) -> list[dict]:
    """生成带 Chroma chunk 元信息的 hybrid 打分调试结果。"""

    result = []

    for hit in hits:
        result.append(
            {
                "preview": preview(
                    hit.get("content", ""),
                    HYBRID_DOC_TEXT_PREVIEW_CHARS,
                ),
                "doc_id": hit.get("doc_id", ""),
                "doc_title": hit.get("doc_title", ""),
                "source": hit.get("source", ""),
                "chunk_index": hit.get("chunk_index", 0),
                "start_char": hit.get("start_char", 0),
                "end_char": hit.get("end_char", 0),
                "chunk_char_len": hit.get("chunk_char_len", 0),
                "retrieval_sources": hit.get("retrieval_sources", []),
                "merged_chunk_ids": hit.get("merged_chunk_ids", []),
                "merged_chunk_indexes": hit.get("merged_chunk_indexes", []),
                "semantic": round(hit.get("semantic_score", 0), 4),
                "keyword": round(hit.get("keyword_score_norm", 0), 4),
                "final": round(hit.get("score", 0), 4),
            }
        )

    return result


def build_rag_debug_payload(
    *,
    rewritten_query: str,
    docs: list[dict],
    filtered_docs: list[dict],
    doc_hits: list[dict],
    merged_doc_hits: list[dict],
    memory_before_rerank: list[dict],
    memory_hits: list[dict],
    doc_retrieval_debug: dict,
    memory_retrieval_debug: dict,
    embedding_profiles: dict,
    stream_used: bool,
    threshold: float,
    doc_context: str,
    citations: list[dict],
    context_compression: dict,
    query_classification: QueryClassification,
    rewrite_result: RewriteResult,
    answer_strategy: dict,
    sub_timings_ms: dict[str, float],
    errors: list[str],
) -> dict:
    """构建写入 debug_info[rag_agent] 的 payload。"""

    return {
        "llm_profiles": {
            "rewrite": get_profile_runtime_info(PROFILE_REWRITE),
            "doc_answer": get_profile_runtime_info(PROFILE_DEFAULT_CHAT),
            "memory_answer": get_profile_runtime_info(PROFILE_DEFAULT_CHAT),
        },
        "embedding_profiles": embedding_profiles,
        "rewritten_query": rewritten_query,
        "doc_used": len(merged_doc_hits) > 0,
        "memory_used": len(memory_hits) > 0,
        "streamed_answer": stream_used,
        "query_type": query_classification.query_type,
        "query_classification": {
            "type": query_classification.query_type,
            "confidence": query_classification.confidence,
            "reason": query_classification.reason,
        },
        "query_rewrite": {
            "mode": rewrite_result.mode,
            "trigger": rewrite_result.trigger,
            "skipped_reason": rewrite_result.skipped_reason,
        },
        "threshold": threshold,
        "doc_context_chars": len(doc_context),
        "context_compression": context_compression,
        "citations": citations,
        "citation_count": len(citations),
        "citation_doc_ids": [
            citation.get("doc_id", "")
            for citation in citations
            if citation.get("doc_id")
        ],
        "answer_strategy": answer_strategy["name"],
        "answer_context_chars": answer_strategy["context_chars"],
        "answer_max_tokens": answer_strategy["max_tokens"],
        "top_docs": preview_doc_hits(docs),
        "filtered_docs": preview_doc_hits(filtered_docs),
        "pre_rerank_docs": preview_doc_hits(filtered_docs),
        "post_rerank_docs": preview_doc_hits(doc_hits),
        "merged_docs": preview_doc_hits(merged_doc_hits),
        "pre_rerank_memory": preview_hits(memory_before_rerank),
        "post_rerank_memory": preview_hits(memory_hits),
        "retrieval_debug": {
            "doc": doc_retrieval_debug,
            "memory": memory_retrieval_debug,
        },
        "sub_timings_ms": sub_timings_ms,
        "errors": errors,
    }


def build_rag_log_extra(
    *,
    docs: list[dict],
    filtered_docs: list[dict],
    doc_hits: list[dict],
    merged_doc_hits: list[dict],
    memory_before_rerank: list[dict],
    memory_hits: list[dict],
    doc_retrieval_debug: dict,
    memory_retrieval_debug: dict,
    embedding_profiles: dict,
    threshold: float,
    context: str,
    doc_context: str,
    citations: list[dict],
    context_compression: dict,
    query_classification: QueryClassification,
    answer_strategy: dict,
    sub_timings_ms: dict[str, float],
    errors: list[str],
) -> dict:
    """构建 log_node 使用的 camelCase 调试字段。"""

    return {
        "threshold": threshold,
        "topDocs": preview_doc_hits(docs),
        "filteredDocs": preview_doc_hits(filtered_docs),
        "memoryHits": preview_hits(memory_hits),
        "docUsed": len(merged_doc_hits) > 0,
        "memoryUsed": len(memory_hits) > 0,
        "queryType": query_classification.query_type,
        "queryClassification": {
            "type": query_classification.query_type,
            "confidence": query_classification.confidence,
            "reason": query_classification.reason,
        },
        "contextPreview": preview(context, 180),
        "docContextChars": len(doc_context),
        "contextCompression": context_compression,
        "citations": citations,
        "citationCount": len(citations),
        "citationDocIds": [
            citation.get("doc_id", "")
            for citation in citations
            if citation.get("doc_id")
        ],
        "answerStrategy": answer_strategy["name"],
        "answerContextChars": answer_strategy["context_chars"],
        "answerMaxTokens": answer_strategy["max_tokens"],
        "preRerankDocs": preview_doc_hits(filtered_docs),
        "postRerankDocs": preview_doc_hits(doc_hits),
        "mergedDocs": preview_doc_hits(merged_doc_hits),
        "preRerankMemory": preview_hits(memory_before_rerank),
        "postRerankMemory": preview_hits(memory_hits),
        "retrievalDebug": {
            "doc": doc_retrieval_debug,
            "memory": memory_retrieval_debug,
        },
        "subTimingsMs": sub_timings_ms,
        "embeddingProfiles": embedding_profiles,
        "errors": errors,
        "hybridDocs": build_hybrid_doc_debug(docs),
    }

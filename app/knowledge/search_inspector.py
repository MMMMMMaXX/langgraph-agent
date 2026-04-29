"""RAG 检索解释工具。

这层只解释“一个 query 在知识库里怎么被检索、过滤、重排、合并和压缩”，
不生成最终答案，也不写 memory。它复用 RAG doc pipeline 的真实 step，
避免出现“调试工具看到的链路”和“线上回答链路”不一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agents.rag.context import build_rag_context
from app.agents.rag.doc_pipeline import (
    build_doc_pipeline_config,
    create_doc_pipeline_state,
    run_chunk_merge_step,
    run_debug_step,
    run_dense_search_step,
    run_hybrid_merge_step,
    run_lexical_search_step,
    run_rerank_step,
    run_source_diversity_step,
    run_threshold_step,
)
from app.agents.rag.query_classifier import classify_rag_query
from app.agents.rag.strategy import build_doc_answer_strategy
from app.agents.rag.types import QueryClassification
from app.config import RAG_CONFIG
from app.utils.logger import preview

DEFAULT_SEARCH_INSPECT_TOP_K = 8
HIT_PREVIEW_CHARS = 140


@dataclass(frozen=True)
class SearchInspectReport:
    """单次 query 的检索解释报告。"""

    query: str
    query_type: str
    query_classification: dict[str, Any]
    pipeline_config: dict[str, Any]
    retrieval_debug: dict[str, Any]
    timings_ms: dict[str, float]
    dense_hits: list[dict]
    lexical_hits: list[dict]
    hybrid_hits: list[dict]
    returned_hits: list[dict]
    filtered_hits: list[dict]
    reranked_hits: list[dict]
    merged_hits: list[dict]
    citations: list[dict]
    context_preview: str
    context_chars: int
    context_compression: dict[str, Any]
    errors: list[str] = field(default_factory=list)


def _summarize_hit(hit: dict) -> dict:
    """把底层 hit 缩成适合 API/CLI 展示的稳定字段。"""

    return {
        "id": hit.get("id", hit.get("chunk_id", "")),
        "doc_id": str(hit.get("doc_id", "")),
        "doc_title": hit.get("doc_title", ""),
        "source": hit.get("source", ""),
        "section_title": hit.get("section_title", ""),
        "chunk_index": hit.get("chunk_index", 0),
        "chunk_char_len": hit.get("chunk_char_len", 0),
        "retrieval_sources": hit.get("retrieval_sources", []),
        "merged_chunk_ids": hit.get("merged_chunk_ids", []),
        "merged_chunk_indexes": hit.get("merged_chunk_indexes", []),
        "score": round(float(hit.get("score", 0.0) or 0.0), 4),
        "semantic_score": round(float(hit.get("semantic_score", 0.0) or 0.0), 4),
        "keyword_score_norm": round(
            float(hit.get("keyword_score_norm", 0.0) or 0.0),
            4,
        ),
        "preview": preview(str(hit.get("content", "")), HIT_PREVIEW_CHARS),
    }


def _summarize_hits(hits: list[dict], limit: int) -> list[dict]:
    return [_summarize_hit(hit) for hit in hits[: max(limit, 0)]]


def _classification_payload(classification: QueryClassification) -> dict[str, Any]:
    return {
        "type": classification.query_type,
        "confidence": classification.confidence,
        "reason": classification.reason,
    }


def _config_payload(config) -> dict[str, Any]:
    return {
        "query_type": config.query_type,
        "doc_top_k": config.doc_top_k,
        "doc_rerank_top_k": config.doc_rerank_top_k,
        "candidate_top_k": config.candidate_top_k,
        "score_threshold": config.score_threshold,
        "soft_match_threshold": config.soft_match_threshold,
        "hybrid_alpha": config.hybrid_alpha,
        "hybrid_beta": config.hybrid_beta,
        "dense_enabled": config.dense_enabled,
        "lexical_enabled": config.lexical_enabled,
        "rerank_enabled": config.rerank_enabled,
        "chunk_merge_enabled": config.chunk_merge_enabled,
        "source_diversity_enabled": config.source_diversity_enabled,
    }


def inspect_retrieval(
    query: str,
    *,
    top_k: int = DEFAULT_SEARCH_INSPECT_TOP_K,
    context_preview_chars: int = RAG_CONFIG.max_doc_context_chars,
) -> SearchInspectReport:
    """执行一次真实文档检索链路，并返回每个阶段的可解释结果。"""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")

    classification = classify_rag_query(
        original_query=normalized_query,
        rewritten_query=normalized_query,
        has_context=False,
    )
    config = build_doc_pipeline_config(
        classification.query_type,
        query=normalized_query,
    )
    state = create_doc_pipeline_state(normalized_query, config)

    for step in (
        run_dense_search_step,
        run_lexical_search_step,
        run_hybrid_merge_step,
        run_threshold_step,
        run_rerank_step,
        run_chunk_merge_step,
        run_source_diversity_step,
        run_debug_step,
    ):
        state = step(state)

    answer_strategy = build_doc_answer_strategy(
        normalized_query,
        classification=classification,
    )
    rag_context = build_rag_context(
        doc_hits=state.merged_doc_hits,
        memory_hits=[],
        doc_context_chars=answer_strategy["context_chars"],
        query=normalized_query,
        query_type=classification.query_type,
    )

    return SearchInspectReport(
        query=normalized_query,
        query_type=classification.query_type,
        query_classification=_classification_payload(classification),
        pipeline_config=_config_payload(config),
        retrieval_debug=state.retrieval_debug,
        timings_ms=state.timings_ms,
        dense_hits=_summarize_hits(state.dense_hits, top_k),
        lexical_hits=_summarize_hits(state.lexical_hits, top_k),
        hybrid_hits=_summarize_hits(state.hybrid_hits, top_k),
        returned_hits=_summarize_hits(state.docs, top_k),
        filtered_hits=_summarize_hits(state.filtered_docs, top_k),
        reranked_hits=_summarize_hits(state.doc_hits, top_k),
        merged_hits=_summarize_hits(state.merged_doc_hits, top_k),
        citations=rag_context.citations,
        context_preview=rag_context.doc_context[: max(context_preview_chars, 0)],
        context_chars=len(rag_context.doc_context),
        context_compression=rag_context.context_compression,
        errors=state.errors,
    )

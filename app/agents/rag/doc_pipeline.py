"""RAG 文档检索管线。

该模块把文档检索拆成可插拔 step：
search -> threshold -> rerank -> chunk merge -> debug/result。
第一版保持外部行为基本不变，但让 query_type 可以逐步驱动不同 pipeline 配置。
"""

from app.agents.rag.chunk_merge import merge_adjacent_doc_hits
from app.agents.rag.constants import (
    QUERY_TYPE_COMPARISON,
    QUERY_TYPE_FALLBACK,
)
from app.agents.rag.doc_policy import should_skip_doc_rerank
from app.agents.rag.types import (
    DocRetrievalPipelineConfig,
    DocRetrievalPipelineState,
    DocRetrievalResult,
)
from app.config import RAG_CONFIG, VECTOR_STORE_CONFIG
from app.rag import search_docs
from app.retrieval.reranker import rerank
from app.utils.errors import build_error_info
from app.utils.logger import now_ms


def build_doc_pipeline_config(query_type: str = "") -> DocRetrievalPipelineConfig:
    """根据 query_type 生成文档检索 pipeline 配置。

    对比类问题通常需要更多候选进入 rerank；fallback 类问题则更保守，避免低置信
    模糊问题硬答。
    """

    doc_top_k = RAG_CONFIG.doc_top_k
    doc_rerank_top_k = RAG_CONFIG.doc_rerank_top_k
    soft_match_threshold = RAG_CONFIG.doc_soft_match_threshold

    if query_type == QUERY_TYPE_COMPARISON:
        doc_top_k = max(doc_top_k, 8)
        doc_rerank_top_k = max(doc_rerank_top_k, 3)
    elif query_type == QUERY_TYPE_FALLBACK:
        soft_match_threshold = RAG_CONFIG.doc_score_threshold

    return DocRetrievalPipelineConfig(
        query_type=query_type or "unknown",
        doc_top_k=doc_top_k,
        doc_rerank_top_k=doc_rerank_top_k,
        score_threshold=RAG_CONFIG.doc_score_threshold,
        soft_match_threshold=soft_match_threshold,
    )


def create_doc_pipeline_state(
    query: str,
    config: DocRetrievalPipelineConfig,
) -> DocRetrievalPipelineState:
    return DocRetrievalPipelineState(
        query=query,
        config=config,
        docs=[],
        filtered_docs=[],
        doc_hits=[],
        merged_doc_hits=[],
        retrieval_debug={},
        errors=[],
        timings_ms={},
    )


def run_search_step(state: DocRetrievalPipelineState) -> DocRetrievalPipelineState:
    """执行 dense/lexical/hybrid 搜索。

    当前底层仍由 search_docs 封装 dense + lexical + hybrid。这里先把 step 边界拉开，
    后续可以把 dense_step / lexical_step / hybrid_merge_step 逐步下沉到这里。
    """

    started_at_ms = now_ms()
    try:
        state.docs = search_docs(state.query, top_k=state.config.doc_top_k)
    except Exception as exc:
        state.docs = []
        state.errors.append(
            build_error_info(
                exc,
                stage="search_docs",
                source="retrieval",
                preferred_code="retrieval_error",
            )
        )
    state.timings_ms["docSearch"] = round(now_ms() - started_at_ms, 2)
    return state


def run_threshold_step(state: DocRetrievalPipelineState) -> DocRetrievalPipelineState:
    """按硬阈值和 soft-match 策略过滤候选。"""

    state.filtered_docs = [
        doc for doc in state.docs if doc["score"] >= state.config.score_threshold
    ]

    if (
        not state.filtered_docs
        and state.docs
        and state.docs[0]["score"] >= state.config.soft_match_threshold
    ):
        state.filtered_docs = state.docs[:1]

    return state


def run_rerank_step(state: DocRetrievalPipelineState) -> DocRetrievalPipelineState:
    """执行 rerank；允许策略决定跳过以节省延迟。"""

    if not state.config.rerank_enabled:
        state.doc_hits = state.filtered_docs[: state.config.doc_rerank_top_k]
        state.retrieval_debug["rerank_skipped"] = True
        state.retrieval_debug["rerank_skip_reason"] = "disabled"
        state.timings_ms["docRerank"] = 0.0
        return state

    skipped, skip_reason = should_skip_doc_rerank(state.filtered_docs)
    state.retrieval_debug["rerank_skipped"] = skipped
    state.retrieval_debug["rerank_skip_reason"] = skip_reason

    if skipped:
        state.doc_hits = state.filtered_docs[: state.config.doc_rerank_top_k]
        state.timings_ms["docRerank"] = 0.0
        return state

    started_at_ms = now_ms()
    try:
        state.doc_hits = rerank(
            state.query,
            state.filtered_docs,
            top_k=state.config.doc_rerank_top_k,
        )
    except Exception as exc:
        state.doc_hits = state.filtered_docs[: state.config.doc_rerank_top_k]
        state.errors.append(
            build_error_info(
                exc,
                stage="doc_rerank",
                source="retrieval",
                preferred_code="retrieval_error",
            )
        )
    state.timings_ms["docRerank"] = round(now_ms() - started_at_ms, 2)
    return state


def run_chunk_merge_step(state: DocRetrievalPipelineState) -> DocRetrievalPipelineState:
    """合并相邻 chunk，生成最终送入回答模型的文档上下文候选。"""

    if not state.config.chunk_merge_enabled:
        state.merged_doc_hits = state.doc_hits[:]
        return state

    state.merged_doc_hits = merge_adjacent_doc_hits(state.doc_hits)
    return state


def run_debug_step(state: DocRetrievalPipelineState) -> DocRetrievalPipelineState:
    """汇总 pipeline debug 指标。"""

    state.retrieval_debug.update(
        {
            "collection": VECTOR_STORE_CONFIG.doc_collection_name,
            "where": None,
            "query_type": state.config.query_type,
            "pipeline_steps": [
                "search",
                "threshold",
                "rerank",
                "chunk_merge",
            ],
            "requested_top_k": state.config.doc_top_k,
            "candidate_top_k": max(
                state.config.doc_top_k * 4,
                state.config.doc_top_k,
            ),
            "returned_count": len(state.docs),
            "filtered_count": len(state.filtered_docs),
            "consumed_count": len(state.doc_hits),
            "merged_count": len(state.merged_doc_hits),
            "merged": len(state.merged_doc_hits) != len(state.doc_hits),
        }
    )
    return state


def build_doc_retrieval_result(
    state: DocRetrievalPipelineState,
) -> DocRetrievalResult:
    return DocRetrievalResult(
        docs=state.docs,
        filtered_docs=state.filtered_docs,
        doc_hits=state.doc_hits,
        merged_doc_hits=state.merged_doc_hits,
        retrieval_debug=state.retrieval_debug,
        errors=state.errors,
        timings_ms=state.timings_ms,
    )


def run_doc_retrieval_pipeline(
    query: str,
    config: DocRetrievalPipelineConfig,
) -> DocRetrievalResult:
    """执行文档检索 pipeline。"""

    state = create_doc_pipeline_state(query, config)
    for step in (
        run_search_step,
        run_threshold_step,
        run_rerank_step,
        run_chunk_merge_step,
        run_debug_step,
    ):
        state = step(state)
    return build_doc_retrieval_result(state)


def retrieve_docs_for_rag(query: str, query_type: str = "") -> DocRetrievalResult:
    """执行 RAG 文档检索主流程。"""

    config = build_doc_pipeline_config(query_type)
    return run_doc_retrieval_pipeline(query, config)

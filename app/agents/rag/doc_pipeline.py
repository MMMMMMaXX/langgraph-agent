"""RAG 文档检索管线。

该模块把文档检索拆成可插拔 step：
dense search -> lexical search -> hybrid merge -> threshold -> rerank
-> chunk merge -> debug/result。
每个召回/融合步骤都有独立开关和 debug 指标，后续替换 BM25、外部搜索引擎
或融合策略时，不需要改 RAG agent 主编排。
"""

from app.agents.rag.chunk_merge import merge_adjacent_doc_hits
from app.agents.rag.constants import (
    QUERY_TYPE_COMPARISON,
    QUERY_TYPE_DEFINITION,
    QUERY_TYPE_FALLBACK,
    QUERY_TYPE_FOLLOWUP,
)
from app.agents.rag.doc_policy import should_skip_doc_rerank
from app.agents.rag.types import (
    DocRetrievalPipelineConfig,
    DocRetrievalPipelineState,
    DocRetrievalResult,
)
from app.config import RAG_CONFIG, VECTOR_STORE_CONFIG
from app.constants.retrieval import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_HYBRID_BETA,
    DOC_CANDIDATE_MULTIPLIER,
)
from app.retrieval.doc_retrieval import (
    apply_keyword_scores,
    dense_retrieve_docs,
    keyword_retrieve_docs,
    merge_doc_hits,
    rank_hybrid,
)
from app.retrieval.reranker import rerank
from app.utils.errors import build_error_info
from app.utils.logger import now_ms

# Hybrid 权重统一维护在这里：alpha=dense semantic，beta=lexical keyword。
# 这些值是检索策略的经验初始值，后续应结合 retrieval eval 指标继续校准。
HYBRID_WEIGHT_BY_QUERY_TYPE: dict[str, tuple[float, float]] = {
    # 定义类通常包含专有名词/缩写，适当提高 lexical，避免 dense 漏掉精确词。
    QUERY_TYPE_DEFINITION: (0.55, 0.45),
    # 对比类需要覆盖多个对象；保留 dense 稳定性，同时增强 lexical 对对象名的约束。
    QUERY_TYPE_COMPARISON: (0.6, 0.4),
    # 追问已先改写成完整 query，语义表达更重要，避免被局部关键词带偏。
    QUERY_TYPE_FOLLOWUP: (0.7, 0.3),
    # 兜底类低置信，dense/lexical 均衡，后续再由 threshold 保守过滤。
    QUERY_TYPE_FALLBACK: (0.5, 0.5),
}
DEFAULT_HYBRID_WEIGHTS = (DEFAULT_HYBRID_ALPHA, DEFAULT_HYBRID_BETA)


def build_query_type_hybrid_weights(
    query_type: str,
    confidence: float = 1.0,
) -> tuple[float, float]:
    """按 query_type 和分类置信度动态调整 dense/lexical 融合权重。

    置信度衰减策略：confidence < 1.0 时，权重向均衡 (0.5, 0.5) 线性收敛。
    公式：alpha = base_alpha * confidence + 0.5 * (1 - confidence)

    示例（FACTUAL 默认 confidence=0.6，base=(0.65, 0.35)）：
      alpha = 0.65 * 0.6 + 0.5 * 0.4 = 0.59，更接近均衡，减少误分类损失。
    高置信度（0.9）几乎不影响 type-specific 权重（偏差 < 0.01）。
    """
    base_alpha, base_beta = HYBRID_WEIGHT_BY_QUERY_TYPE.get(
        query_type, DEFAULT_HYBRID_WEIGHTS
    )
    if confidence >= 1.0:
        return (base_alpha, base_beta)
    alpha = round(base_alpha * confidence + 0.5 * (1.0 - confidence), 4)
    return (alpha, round(1.0 - alpha, 4))


def build_doc_pipeline_config(
    query_type: str = "",
    query: str = "",
    confidence: float = 1.0,
) -> DocRetrievalPipelineConfig:
    """根据 query_type/query 生成文档检索 pipeline 配置。

    对比类问题需要更广覆盖；定义类问题更依赖关键词和专有名词；追问改写后更依赖
    语义召回；fallback 类问题更保守，避免低置信模糊问题硬答。
    confidence 用于衰减 hybrid 权重：分类不确定时向均衡 (0.5, 0.5) 收敛。
    """

    doc_top_k = RAG_CONFIG.doc_top_k
    doc_rerank_top_k = RAG_CONFIG.doc_rerank_top_k
    soft_match_threshold = RAG_CONFIG.doc_soft_match_threshold
    candidate_multiplier = DOC_CANDIDATE_MULTIPLIER
    hybrid_alpha, hybrid_beta = build_query_type_hybrid_weights(query_type, confidence)
    normalized_query = query.strip()

    if query_type == QUERY_TYPE_COMPARISON:
        doc_top_k = max(doc_top_k, 8)
        doc_rerank_top_k = max(doc_rerank_top_k, 4)
        candidate_multiplier = max(candidate_multiplier, 5)
    elif query_type == QUERY_TYPE_DEFINITION:
        doc_top_k = max(doc_top_k, 6)
    elif query_type == QUERY_TYPE_FOLLOWUP:
        doc_top_k = max(doc_top_k, 6)
        doc_rerank_top_k = max(doc_rerank_top_k, 3)
    elif query_type == QUERY_TYPE_FALLBACK:
        soft_match_threshold = RAG_CONFIG.doc_score_threshold

    if normalized_query and len(normalized_query) <= 12:
        doc_top_k = max(doc_top_k, 6)
        candidate_multiplier = max(candidate_multiplier, 5)

    return DocRetrievalPipelineConfig(
        query_type=query_type or "unknown",
        doc_top_k=doc_top_k,
        doc_rerank_top_k=doc_rerank_top_k,
        candidate_top_k=max(doc_top_k * candidate_multiplier, doc_top_k),
        score_threshold=RAG_CONFIG.doc_score_threshold,
        soft_match_threshold=soft_match_threshold,
        hybrid_alpha=hybrid_alpha,
        hybrid_beta=hybrid_beta,
        source_diversity_enabled=query_type == QUERY_TYPE_COMPARISON,
    )


def create_doc_pipeline_state(
    query: str,
    config: DocRetrievalPipelineConfig,
) -> DocRetrievalPipelineState:
    return DocRetrievalPipelineState(
        query=query,
        config=config,
        dense_hits=[],
        lexical_hits=[],
        hybrid_hits=[],
        docs=[],
        filtered_docs=[],
        doc_hits=[],
        merged_doc_hits=[],
        diversified_doc_hits=[],
        retrieval_debug={},
        errors=[],
        timings_ms={},
    )


def run_dense_search_step(
    state: DocRetrievalPipelineState,
) -> DocRetrievalPipelineState:
    """执行 dense 召回。

    dense 负责语义相似召回，适合"表达不完全一致但语义接近"的问题。
    它是高召回入口之一，但不在这里做最终排序。
    """

    if not state.config.dense_enabled:
        state.dense_hits = []
        state.timings_ms["docDenseSearch"] = 0.0
        return state

    started_at_ms = now_ms()
    try:
        state.dense_hits = dense_retrieve_docs(
            state.query,
            top_k=state.config.candidate_top_k,
        )
    except Exception as exc:
        state.dense_hits = []
        state.errors.append(
            build_error_info(
                exc,
                stage="dense_retrieve_docs",
                source="retrieval",
                preferred_code="retrieval_error",
            )
        )
    state.timings_ms["docDenseSearch"] = round(now_ms() - started_at_ms, 2)
    return state


def run_lexical_search_step(
    state: DocRetrievalPipelineState,
) -> DocRetrievalPipelineState:
    """执行 lexical 召回。

    lexical 负责精确词项、英文缩写、专有名词召回。现在底层是 SQLite FTS5，
    以后可替换为更专业的 BM25/ES/OpenSearch 实现。
    """

    if not state.config.lexical_enabled:
        state.lexical_hits = []
        state.timings_ms["docLexicalSearch"] = 0.0
        return state

    started_at_ms = now_ms()
    try:
        state.lexical_hits = keyword_retrieve_docs(
            state.query,
            top_k=state.config.candidate_top_k,
        )
    except Exception as exc:
        state.lexical_hits = []
        state.errors.append(
            build_error_info(
                exc,
                stage="keyword_retrieve_docs",
                source="retrieval",
                preferred_code="retrieval_error",
            )
        )
    state.timings_ms["docLexicalSearch"] = round(now_ms() - started_at_ms, 2)
    return state


def run_hybrid_merge_step(
    state: DocRetrievalPipelineState,
) -> DocRetrievalPipelineState:
    """合并 dense/lexical 结果并计算 hybrid 排序。

    这里是检索融合层：先按 chunk id 去重，再补齐 keyword_score，最后用
    semantic + keyword 的权重融合分排序。最终给下游 threshold 的 `docs`
    仍只保留 doc_top_k，避免 rerank/context 阶段吞太多候选。
    """

    started_at_ms = now_ms()
    hits = merge_doc_hits([state.dense_hits, state.lexical_hits])
    hits = apply_keyword_scores(state.query, hits)
    state.hybrid_hits = rank_hybrid(
        hits,
        alpha=state.config.hybrid_alpha,
        beta=state.config.hybrid_beta,
    )
    state.docs = state.hybrid_hits[: state.config.doc_top_k]
    state.timings_ms["docHybridMerge"] = round(now_ms() - started_at_ms, 2)
    state.timings_ms["docSearch"] = round(
        state.timings_ms.get("docDenseSearch", 0.0)
        + state.timings_ms.get("docLexicalSearch", 0.0)
        + state.timings_ms.get("docHybridMerge", 0.0),
        2,
    )
    return state


def run_search_step(state: DocRetrievalPipelineState) -> DocRetrievalPipelineState:
    """兼容型搜索 step，按 dense -> lexical -> hybrid 顺序执行。"""

    state = run_dense_search_step(state)
    state = run_lexical_search_step(state)
    state = run_hybrid_merge_step(state)
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
        if state.config.query_type == QUERY_TYPE_COMPARISON:
            state.filtered_docs = [
                doc
                for doc in state.docs[: state.config.doc_rerank_top_k]
                if doc["score"] >= state.config.soft_match_threshold
            ]
            return state
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


def select_source_diverse_hits(
    hits: list[dict],
    *,
    max_hits: int,
) -> list[dict]:
    """优先选择不同 doc_id 的上下文块。

    对比类/综合类问题最怕 top blocks 全来自同一文档，导致回答只覆盖一边。
    这里采用轻量两轮策略：第一轮每个 doc_id 取最高分一条，第二轮按原排序补齐。
    """

    if not hits or max_hits <= 0:
        return []

    selected: list[dict] = []
    selected_ids: set[int] = set()
    seen_doc_ids: set[str] = set()

    for index, hit in enumerate(hits):
        doc_id = str(hit.get("doc_id", ""))
        if not doc_id or doc_id in seen_doc_ids:
            continue
        selected.append(hit)
        selected_ids.add(index)
        seen_doc_ids.add(doc_id)
        if len(selected) >= max_hits:
            return selected

    for index, hit in enumerate(hits):
        if index in selected_ids:
            continue
        selected.append(hit)
        if len(selected) >= max_hits:
            break

    return selected


def run_source_diversity_step(
    state: DocRetrievalPipelineState,
) -> DocRetrievalPipelineState:
    """对 comparison query 做来源多样性选择。"""

    if not state.config.source_diversity_enabled:
        state.diversified_doc_hits = state.merged_doc_hits[:]
        state.retrieval_debug["source_diversity_enabled"] = False
        return state

    state.diversified_doc_hits = select_source_diverse_hits(
        state.merged_doc_hits,
        max_hits=RAG_CONFIG.max_doc_context_blocks,
    )
    state.merged_doc_hits = state.diversified_doc_hits[:]
    state.retrieval_debug["source_diversity_enabled"] = True
    state.retrieval_debug["source_diversity_doc_ids"] = [
        hit.get("doc_id", "") for hit in state.diversified_doc_hits
    ]
    return state


def run_debug_step(state: DocRetrievalPipelineState) -> DocRetrievalPipelineState:
    """汇总 pipeline debug 指标。"""

    failed_stages = {
        str(error.get("stage", ""))
        for error in state.errors
        if isinstance(error, dict) and error.get("stage")
    }
    state.retrieval_debug.update(
        {
            "collection": VECTOR_STORE_CONFIG.doc_collection_name,
            "where": None,
            "query_type": state.config.query_type,
            "pipeline_steps": [
                "dense_search",
                "lexical_search",
                "hybrid_merge",
                "threshold",
                "rerank",
                "chunk_merge",
                "source_diversity",
            ],
            "requested_top_k": state.config.doc_top_k,
            "candidate_top_k": state.config.candidate_top_k,
            "dense_enabled": state.config.dense_enabled,
            "lexical_enabled": state.config.lexical_enabled,
            "hybrid_alpha": state.config.hybrid_alpha,
            "hybrid_beta": state.config.hybrid_beta,
            "hybrid_weight_strategy": "query_type_dynamic",
            "dense_count": len(state.dense_hits),
            "lexical_count": len(state.lexical_hits),
            "hybrid_count": len(state.hybrid_hits),
            "returned_count": len(state.docs),
            "filtered_count": len(state.filtered_docs),
            "consumed_count": len(state.doc_hits),
            "merged_count": len(state.merged_doc_hits),
            "merged": len(state.merged_doc_hits) != len(state.doc_hits),
            "source_diverse_count": len(state.diversified_doc_hits),
            "error_count": len(state.errors),
            "dense_failed": "dense_retrieve_docs" in failed_stages,
            "lexical_failed": "keyword_retrieve_docs" in failed_stages,
            "rerank_failed": "doc_rerank" in failed_stages,
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
    return build_doc_retrieval_result(state)


def retrieve_docs_for_rag(
    query: str,
    query_type: str = "",
    confidence: float = 1.0,
) -> DocRetrievalResult:
    """执行 RAG 文档检索主流程。"""

    config = build_doc_pipeline_config(query_type, query=query, confidence=confidence)
    return run_doc_retrieval_pipeline(query, config)

from app.agents.rag.answer import generate_answer_for_context
from app.agents.rag.context import build_rag_context
from app.agents.rag.debug import build_rag_debug_payload, build_rag_log_extra
from app.agents.rag.doc_pipeline import retrieve_docs_for_rag
from app.agents.rag.memory_pipeline import retrieve_memory_for_rag
from app.agents.rag.query_classifier import classify_rag_query
from app.agents.rag.rewrite import rewrite_rag_query
from app.agents.rag.strategy import build_doc_answer_strategy
from app.config import RAG_CONFIG
from app.constants.model_profiles import (
    PROFILE_DOC_EMBEDDING,
    PROFILE_MEMORY_EMBEDDING,
    PROFILE_QUERY_EMBEDDING,
)
from app.constants.routes import ROUTE_RAG_AGENT
from app.llm import get_profile_runtime_info
from app.state import AgentState
from app.streaming import build_answer_streamer
from app.tracing import add_current_run_metadata, build_rag_trace_metadata
from app.utils.logger import log_node


def rag_agent_node(state: AgentState) -> AgentState:
    """RAG Agent 主节点：查询改写 → 文档检索 → 记忆兜底 → 生成答案。

    整体流程分五步：
    1. 查询改写：把用户追问补全为可独立检索的完整问题
    2. 文档检索：到知识库做语义 + 关键词混合检索，得到 merged_doc_hits
    3. 记忆兜底：只有文档不足时，才去向量记忆里捞历史对话
    4. 上下文组装：按策略拼接 doc + memory，得到喂给 LLM 的 prompt 上下文
    5. 答案生成：调 LLM 生成最终回答，必要时走流式

    同时全程记录 timings / errors / debug_info，便于 LangSmith trace 与本地调试。
    """
    # ===== 0. 基础输入与运行时信息 =====
    session_id = state.get("session_id", "default")
    message = state["messages"][-1]["content"]
    messages = state["messages"]
    summary = state.get("summary", "")
    sub_timings_ms: dict[str, float] = {}
    errors: list[str] = []
    # 收集 embedding 的 profile 信息（模型、维度等），只用于 debug 展示，
    # 不影响实际检索，便于在 trace 里对齐"查询侧"和"索引侧"是否使用同一模型。
    embedding_profiles = {
        "doc_query": get_profile_runtime_info(PROFILE_QUERY_EMBEDDING, kind="embedding"),
        "memory_query": get_profile_runtime_info(
            PROFILE_QUERY_EMBEDDING, kind="embedding"
        ),
        "doc_index": get_profile_runtime_info(PROFILE_DOC_EMBEDDING, kind="embedding"),
        "memory_index": get_profile_runtime_info(
            PROFILE_MEMORY_EMBEDDING, kind="embedding"
        ),
    }

    # ===== 1. 查询改写 =====
    # 把"那上海呢？"这种省略型追问，借助历史 messages 和 summary，
    # 改写成"上海气候怎么样？"这种可以独立检索的完整问题。
    rewrite_result = rewrite_rag_query(
        message,
        messages=messages,
        summary=summary,
    )
    rewritten = rewrite_result.query
    errors.extend(rewrite_result.errors)
    sub_timings_ms["rewrite"] = rewrite_result.timing_ms
    query_classification = classify_rag_query(
        original_query=message,
        rewritten_query=rewritten,
        has_context=bool(summary or len(messages) > 1),
    )

    # ===== 2. 文档检索（知识库） =====
    # 语义向量 + 关键词打分混合检索，然后阈值过滤 + 合并同 doc 的多个 chunk。
    # merged_doc_hits 是最终用于生成答案的文档片段集合。
    threshold = RAG_CONFIG.doc_score_threshold
    doc_result = retrieve_docs_for_rag(
        rewritten,
        query_type=query_classification.query_type,
    )
    docs = doc_result.docs
    filtered_docs = doc_result.filtered_docs
    doc_hits = doc_result.doc_hits
    merged_doc_hits = doc_result.merged_doc_hits
    doc_retrieval_debug = doc_result.retrieval_debug
    errors.extend(doc_result.errors)
    sub_timings_ms.update(doc_result.timings_ms)
    # 只要合并后还有文档命中，就认为知识库足以作答，无需再走 memory 兜底。
    has_strong_knowledge = len(merged_doc_hits) > 0

    # ===== 3. 向量记忆兜底 =====
    # 性能优化：文档命中充足时，直接跳过 memory 检索 + rerank，
    # 避免为已经有答案的问题多做一次 Embedding + LLM rerank 调用。
    memory_result = retrieve_memory_for_rag(
        rewritten,
        session_id=session_id,
        enabled=not has_strong_knowledge,
    )
    memory_hits = memory_result.memory_hits
    memory_before_rerank = memory_result.memory_before_rerank
    memory_retrieval_debug = memory_result.retrieval_debug
    errors.extend(memory_result.errors)
    sub_timings_ms.update(memory_result.timings_ms)

    # ===== 4. 构造上下文 =====
    # 策略决定：上下文多长、用什么 max_tokens、对长答案/短答案的约束。
    # 再按策略把 doc + memory 压缩拼装成最终上下文。
    doc_answer_strategy = build_doc_answer_strategy(
        rewritten,
        classification=query_classification,
    )
    rag_context = build_rag_context(
        doc_hits=merged_doc_hits,
        memory_hits=memory_hits,
        doc_context_chars=doc_answer_strategy["context_chars"],
    )

    # ===== 5. 生成答案 =====
    # build_answer_streamer 同时返回 on_delta 回调和共享的 stream_state，
    # 用来把 LLM 增量输出转发到 SSE，并记录"本轮是否真的走了流式"。
    has_memory = len(memory_hits) > 0
    on_delta, stream_state = build_answer_streamer(state, ROUTE_RAG_AGENT)

    # 知识库命中优先；只有文档不足时才使用 memory 兜底。
    answer_result = generate_answer_for_context(
        question=rewritten,
        rag_context=rag_context,
        doc_answer_strategy=doc_answer_strategy,
        has_strong_knowledge=has_strong_knowledge,
        has_memory=has_memory,
        on_delta=on_delta,
    )
    answer = answer_result.answer
    errors.extend(answer_result.errors)
    sub_timings_ms["answerGeneration"] = answer_result.timing_ms

    # ===== 7. 组装 debug 信息（debug 模式下返回给前端） =====
    rag_debug = build_rag_debug_payload(
        rewritten_query=rewritten,
        docs=docs,
        filtered_docs=filtered_docs,
        doc_hits=doc_hits,
        merged_doc_hits=merged_doc_hits,
        memory_before_rerank=memory_before_rerank,
        memory_hits=memory_hits,
        doc_retrieval_debug=doc_retrieval_debug,
        memory_retrieval_debug=memory_retrieval_debug,
        embedding_profiles=embedding_profiles,
        stream_used=stream_state["used"],
        threshold=threshold,
        doc_context=rag_context.doc_context,
        citations=rag_context.citations,
        query_classification=query_classification,
        answer_strategy=doc_answer_strategy,
        sub_timings_ms=sub_timings_ms,
        errors=errors,
    )
    # 把检索指标透传到 LangSmith trace，便于按策略 / 是否命中文档做筛选聚合。
    add_current_run_metadata(
        build_rag_trace_metadata(rag_debug),
        tags=[
            "node:rag_agent",
            f"rag_strategy:{doc_answer_strategy['name']}",
            f"rag_query_type:{query_classification.query_type}",
            f"doc_used:{bool(merged_doc_hits)}",
        ],
        event_name="rag_retrieval_summary",
    )

    # ===== 8. 写入本节点增量 state =====
    next_state: AgentState = {
        "rewritten_query": rewritten,
        "memory_hits": memory_hits,
        "context": rag_context.context,
        "debug_info": {ROUTE_RAG_AGENT: rag_debug},
        "agent_outputs": {ROUTE_RAG_AGENT: answer},
        "answer": answer,
    }
    if stream_state["used"]:
        next_state["streamed_answer"] = True
    log_state = {**state, **next_state}

    # ===== 9. 结构化日志，最后一步，不影响主链路 =====
    log_node(
        ROUTE_RAG_AGENT,
        log_state,
        extra=build_rag_log_extra(
            docs=docs,
            filtered_docs=filtered_docs,
            doc_hits=doc_hits,
            merged_doc_hits=merged_doc_hits,
            memory_before_rerank=memory_before_rerank,
            memory_hits=memory_hits,
            doc_retrieval_debug=doc_retrieval_debug,
            memory_retrieval_debug=memory_retrieval_debug,
            embedding_profiles=embedding_profiles,
            threshold=threshold,
            context=rag_context.context,
            doc_context=rag_context.doc_context,
            citations=rag_context.citations,
            query_classification=query_classification,
            answer_strategy=doc_answer_strategy,
            sub_timings_ms=sub_timings_ms,
            errors=errors,
        ),
    )
    return next_state

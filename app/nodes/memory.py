from app.config import MEMORY_CONFIG
from app.constants.keywords import META_HISTORY_QUERY_KEYWORDS, contains_any
from app.constants.model_profiles import PROFILE_MEMORY_EMBEDDING
from app.constants.policies import (
    SKIP_REASON_DUPLICATE,
    SKIP_REASON_EMPTY_MESSAGE,
    SKIP_REASON_META_QUERY,
)
from app.constants.routes import NODE_MEMORY, ROUTE_CHAT_AGENT
from app.llm import LLMCallError, get_profile_runtime_info, summarize_messages
from app.memory.conversation_history import (
    append_history_event,
    resolve_history_backend,
)
from app.memory.services import (
    build_memory_debug_payload,
    build_memory_log_extra,
    prune_working_messages,
    refresh_summary_if_needed,
    write_history_if_needed,
    write_vector_memory_if_needed,
)
from app.memory.vector_memory import MEMORY_SOURCE_CHAT_ROUND, add_memory_item
from app.memory.write_policy import decide_memory_write
from app.state import AgentState
from app.utils.errors import build_error_info
from app.utils.logger import log_node, now_ms, preview
from app.utils.tags import extract_tags


def should_refresh_summary(messages: list[dict], rewritten_query: str) -> bool:
    if len(messages) < 2:
        return False

    # 摘要更新是高成本操作，默认不要每轮都做。
    # 只有对话足够长，或者用户明确在追问历史/总结时，再触发摘要刷新。
    if len(messages) > MEMORY_CONFIG.summary_trigger:
        return True

    return contains_any(rewritten_query, META_HISTORY_QUERY_KEYWORDS)


def should_skip_summary_refresh(state: AgentState, rewritten_query: str) -> bool:
    routes = state.get("routes", [])

    # 这类“总结/回顾/刚刚/之前”的 meta query，
    # 当前轮答案通常已经由 chat_agent 基于 Working Memory 或长期 memory 直接生成。
    # 即使答案是“资料不足”，同步刷新 summary 也只会增加尾部延迟，
    # 还可能把一次失败的 meta 请求写成新的摘要主题。
    is_meta_query = contains_any(rewritten_query, META_HISTORY_QUERY_KEYWORDS)

    return routes == [ROUTE_CHAT_AGENT] and is_meta_query


def should_skip_history_store(
    user_message: str, rewritten_query: str
) -> tuple[bool, str]:
    """判断当前轮是否应该跳过会话流水写入。

    会话流水用于回放“用户问过什么”。总结/回顾这类 meta query 如果也写进去，
    后续总结就会出现“用户询问总结一下刚才的问题”这种自我污染。
    """

    if not user_message.strip():
        return True, SKIP_REASON_EMPTY_MESSAGE

    query_text = f"{user_message}\n{rewritten_query}"
    if contains_any(query_text, META_HISTORY_QUERY_KEYWORDS):
        return True, SKIP_REASON_META_QUERY

    return False, ""


def memory_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "default")
    messages = state.get("messages", [])
    old_summary = state.get("summary", "")
    answer = state.get("answer", "")
    rewritten_query = state.get("rewritten_query", state["messages"][-1]["content"])
    conversation_history_path = state.get("conversation_history_path", "")
    conversation_history_backend = resolve_history_backend(
        history_path=conversation_history_path or None
    )
    sub_timings_ms: dict[str, float] = {}
    errors: list[str] = []
    embedding_profiles = {
        # 长期记忆写入时的 embedding 配置。
        "vector_store": get_profile_runtime_info(
            PROFILE_MEMORY_EMBEDDING, kind="embedding"
        )
    }

    # 1. 只在“需要压缩上下文”或“用户明确依赖历史摘要”时更新 summary
    summary_result = refresh_summary_if_needed(
        state=state,
        messages=messages,
        old_summary=old_summary,
        rewritten_query=rewritten_query,
        should_skip_summary_refresh=should_skip_summary_refresh,
        should_refresh_summary=should_refresh_summary,
        summarize_messages=summarize_messages,
        build_error_info=build_error_info,
        llm_error_type=LLMCallError,
        now_ms=now_ms,
    )
    new_summary = summary_result.summary
    refreshed_summary = summary_result.refreshed_summary
    skipped_summary_refresh = summary_result.skipped_summary_refresh
    errors.extend(summary_result.errors)
    sub_timings_ms["summaryRefresh"] = summary_result.duration_ms

    user_message = messages[-1]["content"] if len(messages) >= 1 else ""
    memory_write_decision = decide_memory_write(
        state=state,
        user_message=user_message,
        rewritten_query=rewritten_query,
        answer=answer,
    )

    # 2. 写入 Vector Memory（只写有价值的结论）
    vector_store_result = write_vector_memory_if_needed(
        session_id=session_id,
        user_message=user_message,
        rewritten_query=rewritten_query,
        answer=answer,
        memory_write_decision=memory_write_decision,
        memory_source=MEMORY_SOURCE_CHAT_ROUND,
        add_memory_item=add_memory_item,
        build_error_info=build_error_info,
        now_ms=now_ms,
    )
    stored_to_vector = vector_store_result.stored_to_vector
    skipped_vector_store = vector_store_result.skipped_vector_store
    vector_store_skip_reason = vector_store_result.vector_store_skip_reason
    stored_tags = vector_store_result.stored_tags
    stored_preview = vector_store_result.stored_preview
    errors.extend(vector_store_result.errors)
    sub_timings_ms["vectorStore"] = vector_store_result.duration_ms

    # 3. 写入非向量化会话流水。
    # 这条流水只服务“总结/回放”，即使当前轮跳过 vector memory，也可以保留问题顺序。
    history_store_result = write_history_if_needed(
        session_id=session_id,
        user_message=user_message,
        rewritten_query=rewritten_query,
        answer=answer,
        routes=state.get("routes", []),
        tags=stored_tags or list(memory_write_decision.tags) or extract_tags(rewritten_query),
        stored_to_vector=stored_to_vector,
        skipped_vector_store=skipped_vector_store,
        vector_store_skip_reason=vector_store_skip_reason,
        history_path=conversation_history_path,
        should_skip_history_store=should_skip_history_store,
        append_history_event=append_history_event,
        duplicate_skip_reason=SKIP_REASON_DUPLICATE,
        build_error_info=build_error_info,
        now_ms=now_ms,
    )
    stored_to_history = history_store_result.stored_to_history
    skipped_history_store = history_store_result.skipped_history_store
    history_store_skip_reason = history_store_result.history_store_skip_reason
    history_preview = history_store_result.history_preview
    errors.extend(history_store_result.errors)
    sub_timings_ms["historyStore"] = history_store_result.duration_ms

    # 4. 如果 messages 太长，再裁剪  Working Memory
    prune_result = prune_working_messages(messages, now_ms=now_ms)
    new_messages = prune_result.messages
    sub_timings_ms["messagePrune"] = prune_result.duration_ms

    new_state = {
        "summary": new_summary,
        "messages": new_messages,
    }
    new_state["debug_info"] = {
        NODE_MEMORY: build_memory_debug_payload(
            refreshed_summary=refreshed_summary,
            skipped_summary_refresh=skipped_summary_refresh,
            stored_to_vector=stored_to_vector,
            skipped_vector_store=skipped_vector_store,
            vector_store_skip_reason=vector_store_skip_reason,
            stored_to_history=stored_to_history,
            skipped_history_store=skipped_history_store,
            history_store_skip_reason=history_store_skip_reason,
            conversation_history_path=conversation_history_path,
            conversation_history_backend=conversation_history_backend,
            memory_write_decision=memory_write_decision,
            stored_tags=stored_tags,
            summary=new_summary,
            sub_timings_ms=sub_timings_ms,
            embedding_profiles=embedding_profiles,
            errors=errors,
            preview=preview,
        )
    }
    log_state = {**state, **new_state}

    log_node(
        NODE_MEMORY,
        log_state,
        extra=build_memory_log_extra(
            refreshed_summary=refreshed_summary,
            skipped_summary_refresh=skipped_summary_refresh,
            stored_to_vector=stored_to_vector,
            skipped_vector_store=skipped_vector_store,
            vector_store_skip_reason=vector_store_skip_reason,
            stored_to_history=stored_to_history,
            skipped_history_store=skipped_history_store,
            history_store_skip_reason=history_store_skip_reason,
            conversation_history_path=conversation_history_path,
            conversation_history_backend=conversation_history_backend,
            memory_write_decision=memory_write_decision,
            stored_tags=stored_tags,
            stored_preview=stored_preview,
            history_preview=history_preview,
            summary=new_summary,
            sub_timings_ms=sub_timings_ms,
            embedding_profiles=embedding_profiles,
            errors=errors,
            preview=preview,
        ),
    )
    return new_state

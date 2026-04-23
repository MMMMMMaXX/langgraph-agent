from app.constants.keywords import META_HISTORY_QUERY_KEYWORDS, contains_any
from app.constants.policies import (
    SKIP_REASON_DUPLICATE,
    SKIP_REASON_EMPTY_MESSAGE,
    SKIP_REASON_META_QUERY,
)
from app.constants.model_profiles import PROFILE_MEMORY_EMBEDDING
from app.constants.routes import NODE_MEMORY, ROUTE_CHAT_AGENT
from app.state import AgentState
from app.config import MEMORY_CONFIG
from app.llm import summarize_messages, LLMCallError, get_profile_runtime_info
from app.utils.errors import build_error_info
from app.utils.logger import log_node, now_ms, preview
from app.memory.vector_memory import MEMORY_SOURCE_CHAT_ROUND, add_memory_item
from app.memory.write_policy import decide_memory_write
from app.memory.conversation_history import (
    append_history_event,
    resolve_history_backend,
)
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
    new_summary = old_summary
    refreshed_summary = False
    skipped_summary_refresh = False
    summary_refresh_started_at_ms = now_ms()
    if should_skip_summary_refresh(state, rewritten_query):
        skipped_summary_refresh = True
    elif should_refresh_summary(messages, rewritten_query):
        try:
            recent_chunk = messages[-MEMORY_CONFIG.max_recent_messages :]
            new_summary = summarize_messages(old_summary, recent_chunk)
            refreshed_summary = True
        except LLMCallError as exc:
            errors.append(
                build_error_info(exc, stage="summarize_messages", source="llm")
            )
    sub_timings_ms["summaryRefresh"] = round(
        now_ms() - summary_refresh_started_at_ms, 2
    )

    stored_to_vector = False
    stored_tags: list[str] = []
    stored_preview = ""
    user_message = messages[-1]["content"] if len(messages) >= 1 else ""
    memory_write_decision = decide_memory_write(
        state=state,
        user_message=user_message,
        rewritten_query=rewritten_query,
        answer=answer,
    )
    skipped_vector_store = not memory_write_decision.should_write
    vector_store_skip_reason = memory_write_decision.skip_reason

    # 2. 写入 Vector Memory（只写有价值的结论）
    vector_store_started_at_ms = now_ms()
    if memory_write_decision.should_write and len(messages) >= 1:
        stored_tags = list(memory_write_decision.tags)
        memory_text = f"""
问题：{user_message}
重写问题：{rewritten_query}
回答：{answer}
标签：{",".join(stored_tags)}
"""
        try:
            add_memory_item(
                memory_text,
                source=MEMORY_SOURCE_CHAT_ROUND,
                rewritten_query=rewritten_query,
                session_id=session_id,
                source_route=memory_write_decision.source_route,
                confidence=memory_write_decision.confidence,
                tags=stored_tags,
                memory_key=memory_write_decision.memory_key,
                memory_type=memory_write_decision.memory_type,
            )
            stored_to_vector = True
            stored_preview = memory_text
        except Exception as exc:
            errors.append(
                build_error_info(
                    exc,
                    stage="add_memory_item",
                    source="memory",
                    preferred_code="storage_error",
                )
            )
    sub_timings_ms["vectorStore"] = round(now_ms() - vector_store_started_at_ms, 2)

    # 3. 写入非向量化会话流水。
    # 这条流水只服务“总结/回放”，即使当前轮跳过 vector memory，也可以保留问题顺序。
    stored_to_history = False
    skipped_history_store = False
    history_store_skip_reason = ""
    history_preview = ""
    history_store_started_at_ms = now_ms()
    skipped_history_store, history_store_skip_reason = should_skip_history_store(
        user_message,
        rewritten_query,
    )
    if not skipped_history_store:
        try:
            history_event = append_history_event(
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
            )
            if history_event.get("skipped_duplicate"):
                skipped_history_store = True
                history_store_skip_reason = SKIP_REASON_DUPLICATE
            else:
                stored_to_history = True
                history_preview = history_event.get(
                    "rewritten_query"
                ) or history_event.get("user_message", "")
        except Exception as exc:
            errors.append(
                build_error_info(
                    exc,
                    stage="append_history_event",
                    source="memory",
                    preferred_code="storage_error",
                )
            )
    sub_timings_ms["historyStore"] = round(now_ms() - history_store_started_at_ms, 2)

    # 4. 如果 messages 太长，再裁剪  Working Memory
    message_prune_started_at_ms = now_ms()
    new_messages = messages
    if len(messages) > MEMORY_CONFIG.summary_trigger:
        new_messages = messages[-MEMORY_CONFIG.max_recent_messages :]
    sub_timings_ms["messagePrune"] = round(now_ms() - message_prune_started_at_ms, 2)

    new_state = {
        "summary": new_summary,
        "messages": new_messages,
    }
    new_state["debug_info"] = {
        NODE_MEMORY: {
            "refreshed_summary": refreshed_summary,
            "skipped_summary_refresh": skipped_summary_refresh,
            "stored_to_vector": stored_to_vector,
            "skipped_vector_store": skipped_vector_store,
            "vector_store_skip_reason": vector_store_skip_reason,
            "stored_to_history": stored_to_history,
            "skipped_history_store": skipped_history_store,
            "history_store_skip_reason": history_store_skip_reason,
            "conversation_history_path": conversation_history_path,
            "conversation_history_backend": conversation_history_backend,
            "memory_write_decision": memory_write_decision.to_debug_dict(),
            "stored_tags": stored_tags,
            "summary_preview": preview(new_summary, 160),
            "sub_timings_ms": sub_timings_ms,
            "embedding_profiles": embedding_profiles,
            "errors": errors,
        }
    }
    log_state = {**state, **new_state}

    log_node(
        NODE_MEMORY,
        log_state,
        extra={
            "refreshedSummary": refreshed_summary,
            "skippedSummaryRefresh": skipped_summary_refresh,
            "storedToVector": stored_to_vector,
            "skippedVectorStore": skipped_vector_store,
            "vectorStoreSkipReason": vector_store_skip_reason,
            "storedToHistory": stored_to_history,
            "skippedHistoryStore": skipped_history_store,
            "historyStoreSkipReason": history_store_skip_reason,
            "conversationHistoryPath": conversation_history_path,
            "conversationHistoryBackend": conversation_history_backend,
            "memoryWriteDecision": memory_write_decision.to_debug_dict(),
            "storedTags": stored_tags,
            "storedPreview": preview(stored_preview, 120),
            "historyPreview": preview(history_preview, 120),
            "summaryPreview": preview(new_summary, 160),
            "subTimingsMs": sub_timings_ms,
            "embeddingProfiles": embedding_profiles,
            "errors": errors,
        },
    )
    return new_state

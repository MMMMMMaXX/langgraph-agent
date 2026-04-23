from app.config import CONVERSATION_HISTORY_CONFIG, VECTOR_STORE_CONFIG
from app.constants.model_profiles import PROFILE_DEFAULT_CHAT, PROFILE_SUMMARY
from app.constants.policies import MEMORY_POLICY_WORKING_ONLY
from app.llm import get_profile_runtime_info
from app.memory.conversation_history import preview_history_events
from app.utils.logger import preview_hits


def build_retrieval_debug(
    *,
    session_id: str,
    memory_lookup_policy: str,
    history_lookup_policy: str,
    conversation_history_backend: str,
    conversation_history_path: str,
    memory_before_rerank: list[dict],
    memory_hits: list[dict],
    working_memory_items: list[str],
    history_events: list[dict],
    history_items: list[str],
    used_memory: bool,
    used_history: bool,
) -> dict:
    """构建 chat_agent 的 retrieval_debug，保持 API debug 字段稳定。"""

    return {
        "memory": {
            "policy": memory_lookup_policy,
            "collection": (
                ""
                if memory_lookup_policy == MEMORY_POLICY_WORKING_ONLY
                else VECTOR_STORE_CONFIG.memory_collection_name
            ),
            "where": (
                None
                if memory_lookup_policy == MEMORY_POLICY_WORKING_ONLY
                else {"session_id": session_id}
            ),
            "returned_count": len(memory_before_rerank),
            "consumed_count": len(memory_hits) if memory_hits and used_memory else 0,
            "working_memory_item_count": len(working_memory_items),
        },
        "history": {
            "backend": conversation_history_backend,
            "policy": history_lookup_policy,
            "path": CONVERSATION_HISTORY_CONFIG.path,
            "effective_path": conversation_history_path
            or CONVERSATION_HISTORY_CONFIG.path,
            "returned_count": len(history_events),
            "consumed_count": len(history_items) if used_history else 0,
        },
    }


def build_chat_debug_payload(
    *,
    task: str,
    memory_lookup_policy: str,
    history_lookup_policy: str,
    used_memory: bool,
    used_history: bool,
    used_summary: bool,
    streamed_answer: bool,
    memory_before_rerank: list[dict],
    memory_hits: list[dict],
    history_events: list[dict],
    retrieval_debug: dict,
    facts: dict,
    sub_timings_ms: dict[str, float],
    errors: list[str],
) -> dict:
    """构建写入 debug_info[chat_agent] 的 payload。"""

    return {
        "llm_profiles": {
            "answer": get_profile_runtime_info(PROFILE_DEFAULT_CHAT),
            "summary": get_profile_runtime_info(PROFILE_SUMMARY),
        },
        "task_type": task,
        "memory_lookup_policy": memory_lookup_policy,
        "history_lookup_policy": history_lookup_policy,
        "used_memory": used_memory,
        "used_history": used_history,
        "used_summary": used_summary,
        "streamed_answer": streamed_answer,
        "pre_rerank_memory": preview_hits(memory_before_rerank),
        "post_rerank_memory": preview_hits(memory_hits),
        "history_hits": preview_history_events(history_events),
        "retrieval_debug": retrieval_debug,
        "facts": facts,
        "sub_timings_ms": sub_timings_ms,
        "errors": errors,
    }


def build_chat_log_extra(
    *,
    task: str,
    memory_lookup_policy: str,
    history_lookup_policy: str,
    used_memory: bool,
    used_history: bool,
    used_summary: bool,
    memory_before_rerank: list[dict],
    memory_hits: list[dict],
    history_events: list[dict],
    retrieval_debug: dict,
    facts: dict,
    sub_timings_ms: dict[str, float],
    errors: list[str],
) -> dict:
    """构建 log_node 使用的 camelCase 调试字段。"""

    return {
        "taskType": task,
        "memoryLookupPolicy": memory_lookup_policy,
        "historyLookupPolicy": history_lookup_policy,
        "usedMemory": used_memory,
        "usedHistory": used_history,
        "usedSummary": used_summary,
        "memoryHits": preview_hits(memory_hits),
        "preRerankMemory": preview_hits(memory_before_rerank),
        "postRerankMemory": preview_hits(memory_hits),
        "historyHits": preview_history_events(history_events),
        "retrievalDebug": retrieval_debug,
        "facts": facts,
        "subTimingsMs": sub_timings_ms,
        "errors": errors,
    }

from __future__ import annotations

from typing import Any

from app.memory.services.types import PreviewFn
from app.memory.write_policy import MemoryWriteDecision


def build_memory_debug_payload(
    *,
    refreshed_summary: bool,
    skipped_summary_refresh: bool,
    stored_to_vector: bool,
    skipped_vector_store: bool,
    vector_store_skip_reason: str,
    stored_to_history: bool,
    skipped_history_store: bool,
    history_store_skip_reason: str,
    conversation_history_path: str,
    conversation_history_backend: str,
    memory_write_decision: MemoryWriteDecision,
    stored_tags: list[str],
    summary: str,
    sub_timings_ms: dict[str, float],
    embedding_profiles: dict[str, dict],
    errors: list[dict],
    preview: PreviewFn,
) -> dict[str, Any]:
    """构造写回 state 的 debug_info[NODE_MEMORY] payload。"""

    return {
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
        "summary_preview": preview(summary, 160),
        "sub_timings_ms": sub_timings_ms,
        "embedding_profiles": embedding_profiles,
        "errors": errors,
    }


def build_memory_log_extra(
    *,
    refreshed_summary: bool,
    skipped_summary_refresh: bool,
    stored_to_vector: bool,
    skipped_vector_store: bool,
    vector_store_skip_reason: str,
    stored_to_history: bool,
    skipped_history_store: bool,
    history_store_skip_reason: str,
    conversation_history_path: str,
    conversation_history_backend: str,
    memory_write_decision: MemoryWriteDecision,
    stored_tags: list[str],
    stored_preview: str,
    history_preview: str,
    summary: str,
    sub_timings_ms: dict[str, float],
    embedding_profiles: dict[str, dict],
    errors: list[dict],
    preview: PreviewFn,
) -> dict[str, Any]:
    """构造 log_node(extra=...) 的 memory 调试字段。"""

    return {
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
        "summaryPreview": preview(summary, 160),
        "subTimingsMs": sub_timings_ms,
        "embeddingProfiles": embedding_profiles,
        "errors": errors,
    }

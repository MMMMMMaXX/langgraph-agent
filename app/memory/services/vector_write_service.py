from __future__ import annotations

from collections.abc import Callable

from app.memory.services.types import ErrorBuilder, VectorStoreResult
from app.memory.write_policy import MemoryWriteDecision


def write_vector_memory_if_needed(
    *,
    session_id: str,
    user_message: str,
    rewritten_query: str,
    answer: str,
    memory_write_decision: MemoryWriteDecision,
    memory_source: str,
    add_memory_item: Callable[..., None],
    build_error_info: ErrorBuilder,
    now_ms: Callable[[], float],
) -> VectorStoreResult:
    """按决策结果写入 Chroma semantic memory。"""

    started_at_ms = now_ms()
    stored_to_vector = False
    stored_tags = list(memory_write_decision.tags)
    stored_preview = ""
    errors: list[dict] = []

    skipped_vector_store = not memory_write_decision.should_write
    vector_store_skip_reason = memory_write_decision.skip_reason

    if memory_write_decision.should_write and user_message:
        memory_text = f"""
问题：{user_message}
重写问题：{rewritten_query}
回答：{answer}
标签：{",".join(stored_tags)}
"""
        try:
            add_memory_item(
                memory_text,
                source=memory_source,
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

    return VectorStoreResult(
        stored_to_vector=stored_to_vector,
        skipped_vector_store=skipped_vector_store,
        vector_store_skip_reason=vector_store_skip_reason,
        stored_tags=stored_tags,
        stored_preview=stored_preview,
        errors=errors,
        duration_ms=round(now_ms() - started_at_ms, 2),
    )

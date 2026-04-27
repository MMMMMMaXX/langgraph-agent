from __future__ import annotations

from collections.abc import Callable

from app.memory.services.types import ErrorBuilder, HistoryStoreResult


def write_history_if_needed(
    *,
    session_id: str,
    user_message: str,
    rewritten_query: str,
    answer: str,
    routes: list[str],
    tags: list[str],
    stored_to_vector: bool,
    skipped_vector_store: bool,
    vector_store_skip_reason: str,
    history_path: str,
    should_skip_history_store: Callable[[str, str], tuple[bool, str]],
    append_history_event: Callable[..., dict],
    duplicate_skip_reason: str,
    build_error_info: ErrorBuilder,
    now_ms: Callable[[], float],
) -> HistoryStoreResult:
    """按策略写入非向量化会话流水。"""

    started_at_ms = now_ms()
    stored_to_history = False
    history_preview = ""
    errors: list[dict] = []

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
                routes=routes,
                tags=tags,
                stored_to_vector=stored_to_vector,
                skipped_vector_store=skipped_vector_store,
                vector_store_skip_reason=vector_store_skip_reason,
                history_path=history_path,
            )
            if history_event.get("skipped_duplicate"):
                skipped_history_store = True
                history_store_skip_reason = duplicate_skip_reason
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

    return HistoryStoreResult(
        stored_to_history=stored_to_history,
        skipped_history_store=skipped_history_store,
        history_store_skip_reason=history_store_skip_reason,
        history_preview=history_preview,
        errors=errors,
        duration_ms=round(now_ms() - started_at_ms, 2),
    )

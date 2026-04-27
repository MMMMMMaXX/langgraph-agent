from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.config import MEMORY_CONFIG
from app.memory.services.types import ErrorBuilder, SummaryRefreshResult


def refresh_summary_if_needed(
    *,
    state: dict[str, Any],
    messages: list[dict],
    old_summary: str,
    rewritten_query: str,
    should_skip_summary_refresh: Callable[[dict[str, Any], str], bool],
    should_refresh_summary: Callable[[list[dict], str], bool],
    summarize_messages: Callable[[str, list[dict]], str],
    build_error_info: ErrorBuilder,
    llm_error_type: type[Exception],
    now_ms: Callable[[], float],
) -> SummaryRefreshResult:
    """按策略决定是否刷新 summary。"""

    new_summary = old_summary
    refreshed_summary = False
    skipped_summary_refresh = False
    errors: list[dict] = []

    started_at_ms = now_ms()
    if should_skip_summary_refresh(state, rewritten_query):
        skipped_summary_refresh = True
    elif should_refresh_summary(messages, rewritten_query):
        try:
            recent_chunk = messages[-MEMORY_CONFIG.max_recent_messages :]
            new_summary = summarize_messages(old_summary, recent_chunk)
            refreshed_summary = True
        except llm_error_type as exc:
            errors.append(
                build_error_info(exc, stage="summarize_messages", source="llm")
            )

    return SummaryRefreshResult(
        summary=new_summary,
        refreshed_summary=refreshed_summary,
        skipped_summary_refresh=skipped_summary_refresh,
        errors=errors,
        duration_ms=round(now_ms() - started_at_ms, 2),
    )

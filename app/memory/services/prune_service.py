from __future__ import annotations

from collections.abc import Callable

from app.config import MEMORY_CONFIG
from app.memory.services.types import MessagePruneResult


def prune_working_messages(
    messages: list[dict],
    *,
    now_ms: Callable[[], float],
) -> MessagePruneResult:
    """按 working memory 上限裁剪消息窗口。"""

    started_at_ms = now_ms()
    new_messages = messages
    if len(messages) > MEMORY_CONFIG.summary_trigger:
        new_messages = messages[-MEMORY_CONFIG.max_recent_messages :]
    return MessagePruneResult(
        messages=new_messages,
        duration_ms=round(now_ms() - started_at_ms, 2),
    )

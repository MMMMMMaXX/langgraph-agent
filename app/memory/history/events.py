"""History event 的构造与内存级去重。

这一层只和 event dict 打交道，不接触任何持久化细节。
SQLite/JSONL backend 都会复用 `build_history_event`，保证不同存储下
字段结构一致，只在“写到哪里”这一步发生分支。
"""

from __future__ import annotations

from typing import Any

from app.utils.logger import now_timestamp_s, preview

from .schema import (
    ANSWER_PREVIEW_CHARS,
    DEFAULT_HISTORY_SOURCE,
    HISTORY_EVENT_VERSION,
    normalize_history_question,
    normalize_routes,
    normalize_tags,
)


def build_history_event(
    *,
    session_id: str,
    user_message: str,
    answer: str,
    rewritten_query: str = "",
    routes: list[str] | None = None,
    tags: list[str] | None = None,
    stored_to_vector: bool = False,
    skipped_vector_store: bool = False,
    vector_store_skip_reason: str = "",
    source: str = DEFAULT_HISTORY_SOURCE,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """把主链路状态整理成统一 history event。

    不论底层写 SQLite 还是 JSONL，业务层都先构造同一份 dict。
    这样后端切换不会影响 memory_node / chat_agent 的调用方式。
    """

    current_time = now_timestamp_s() if timestamp is None else timestamp
    return {
        "version": HISTORY_EVENT_VERSION,
        "session_id": session_id,
        "timestamp": current_time,
        "source": source,
        "routes": routes or [],
        "user_message": user_message,
        "rewritten_query": rewritten_query,
        "normalized_question": normalize_history_question(
            user_message, rewritten_query
        ),
        "answer_preview": preview(answer, ANSWER_PREVIEW_CHARS),
        "tags": normalize_tags(tags),
        "stored_to_vector": stored_to_vector,
        "skipped_vector_store": skipped_vector_store,
        "vector_store_skip_reason": vector_store_skip_reason,
        "skipped_duplicate": False,
    }


def make_dedupe_stub(event: dict[str, Any]) -> dict[str, Any]:
    """构造去重命中时的返回值。

    上层只关心 skipped_duplicate 标记和基本会话上下文，不需要完整
    event 字段。不同后端返回一致的 stub，方便调用方做统一处理。
    """

    return {
        "session_id": event.get("session_id", ""),
        "user_message": event.get("user_message", ""),
        "rewritten_query": event.get("rewritten_query", ""),
        "skipped_duplicate": True,
    }


def is_duplicate_in_memory(
    *,
    events: list[dict[str, Any]],
    session_id: str,
    user_message: str,
    rewritten_query: str = "",
    routes: list[str] | None = None,
    now_timestamp: float | None = None,
    window_seconds: int,
) -> bool:
    """在一批内存 event 上做窗口内去重判断。

    去重只看“同一 session 下同一路由的同一个标准化问题”，不比较答案内容。
    这样可以抑制重复验证产生的噪音，又不会吞掉不同问题或不同 agent 的记录。

    该函数主要服务 JSONL backend（它必须先把文件读进内存）和治理脚本，
    SQLite backend 走索引直接在数据库里判重。
    """

    if window_seconds <= 0:
        return False

    current_time = now_timestamp_s() if now_timestamp is None else now_timestamp
    question = normalize_history_question(user_message, rewritten_query)
    normalized_routes = normalize_routes(routes)

    for event in reversed(events):
        if event.get("session_id") != session_id:
            continue
        event_timestamp = float(event.get("timestamp") or 0)
        if current_time - event_timestamp > window_seconds:
            break

        event_question = normalize_history_question(
            str(event.get("user_message") or ""),
            str(event.get("rewritten_query") or ""),
        )
        if event_question != question:
            continue
        if normalize_routes(event.get("routes") or []) != normalized_routes:
            continue
        return True

    return False

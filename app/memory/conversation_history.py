"""兼容门面：将旧的 `app.memory.conversation_history` 模块入口转发到
拆分后的 `app.memory.history` 包。

历史上所有调用方（api / agents / scripts）都 `from app.memory.conversation_history
import X`，拆分后保留这个模块只做 re-export，避免一次性大改导入点。
新代码建议直接使用 `from app.memory.history import X`。
"""

from __future__ import annotations

from app.memory.history import (  # noqa: F401  - re-export for backward compat
    ANSWER_PREVIEW_CHARS,
    DEFAULT_DEDUPE_ENABLED,
    DEFAULT_HISTORY_SOURCE,
    HISTORY_BACKEND_JSONL,
    HISTORY_BACKEND_SQLITE,
    HISTORY_EVENT_VERSION,
    HISTORY_PREVIEW_CHARS,
    JSONL_SUFFIX,
    SQLITE_SUFFIXES,
    HistoryBackend,
    JsonlBackend,
    SQLiteBackend,
    append_history_event,
    build_history_event,
    get_all_history,
    get_recent_history,
    is_duplicate_in_memory,
    json_dumps,
    json_loads_list,
    make_dedupe_stub,
    normalize_history_question,
    normalize_routes,
    normalize_tags,
    preview_history_events,
    read_history_events,
    resolve_history_backend,
    resolve_history_path,
    write_history_events,
)

# 保留旧名字方便零改动的历史调用方；新代码应使用 is_duplicate_in_memory。
is_duplicate_history_event = is_duplicate_in_memory

__all__ = [
    "ANSWER_PREVIEW_CHARS",
    "DEFAULT_DEDUPE_ENABLED",
    "DEFAULT_HISTORY_SOURCE",
    "HISTORY_BACKEND_JSONL",
    "HISTORY_BACKEND_SQLITE",
    "HISTORY_EVENT_VERSION",
    "HISTORY_PREVIEW_CHARS",
    "HistoryBackend",
    "JsonlBackend",
    "JSONL_SUFFIX",
    "SQLITE_SUFFIXES",
    "SQLiteBackend",
    "append_history_event",
    "build_history_event",
    "get_all_history",
    "get_recent_history",
    "is_duplicate_history_event",
    "is_duplicate_in_memory",
    "json_dumps",
    "json_loads_list",
    "make_dedupe_stub",
    "normalize_history_question",
    "normalize_routes",
    "normalize_tags",
    "preview_history_events",
    "read_history_events",
    "resolve_history_backend",
    "resolve_history_path",
    "write_history_events",
]

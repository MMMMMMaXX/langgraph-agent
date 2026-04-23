"""history 包的公共入口。

这里只做 re-export，让 `from app.memory.history import X` 和
旧的 `from app.memory.conversation_history import X` 拿到同一份对象。

分层：
- schema      常量 / DDL / 规范化工具
- events      event 构造 + 内存版判重
- backend     HistoryBackend Protocol
- sqlite/jsonl_backend  两个具体实现
- service     resolver + 对外高层 API（这一层负责串起其它模块）
"""

from __future__ import annotations

from .backend import HistoryBackend
from .events import (
    build_history_event,
    is_duplicate_in_memory,
    make_dedupe_stub,
)
from .jsonl_backend import JsonlBackend
from .schema import (
    ANSWER_PREVIEW_CHARS,
    DEFAULT_DEDUPE_ENABLED,
    DEFAULT_HISTORY_SOURCE,
    HISTORY_BACKEND_JSONL,
    HISTORY_BACKEND_SQLITE,
    HISTORY_EVENT_VERSION,
    HISTORY_PREVIEW_CHARS,
    JSONL_SUFFIX,
    SQLITE_SUFFIXES,
    json_dumps,
    json_loads_list,
    normalize_history_question,
    normalize_routes,
    normalize_tags,
)
from .service import (
    append_history_event,
    get_all_history,
    get_recent_history,
    preview_history_events,
    read_history_events,
    resolve_history_backend,
    resolve_history_path,
    write_history_events,
)
from .sqlite_backend import SQLiteBackend

__all__ = [
    # backend protocol & impls
    "HistoryBackend",
    "SQLiteBackend",
    "JsonlBackend",
    # events
    "build_history_event",
    "is_duplicate_in_memory",
    "make_dedupe_stub",
    # schema helpers & constants
    "ANSWER_PREVIEW_CHARS",
    "DEFAULT_DEDUPE_ENABLED",
    "DEFAULT_HISTORY_SOURCE",
    "HISTORY_BACKEND_JSONL",
    "HISTORY_BACKEND_SQLITE",
    "HISTORY_EVENT_VERSION",
    "HISTORY_PREVIEW_CHARS",
    "JSONL_SUFFIX",
    "SQLITE_SUFFIXES",
    "json_dumps",
    "json_loads_list",
    "normalize_history_question",
    "normalize_routes",
    "normalize_tags",
    # service / public API
    "append_history_event",
    "get_all_history",
    "get_recent_history",
    "preview_history_events",
    "read_history_events",
    "resolve_history_backend",
    "resolve_history_path",
    "write_history_events",
]

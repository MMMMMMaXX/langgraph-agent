"""Service 层：后端解析 + 统一的 high-level API。

主链路（api / chat / memory_node）和治理脚本都应该调用这里的函数，
不直接依赖 sqlite_backend / jsonl_backend，这样未来再加后端只需扩展 resolver。

关键职责：
- resolve_history_backend / resolve_history_path：
  按“显式 backend > 请求级路径后缀 > 全局配置”的优先级挑选后端与文件。
- _get_backend：
  根据后端名惰性构造 backend 实例，并在构造时注入 dedupe_window_seconds，
  避免 backend 直接读全局配置。
- append/read/write/preview：
  对外暴露与旧 conversation_history.py 兼容的函数签名，
  保证迁移期间所有调用方不用改动。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import CONVERSATION_HISTORY_CONFIG
from app.utils.logger import preview

from .backend import HistoryBackend
from .events import build_history_event
from .jsonl_backend import JsonlBackend
from .schema import (
    DEFAULT_DEDUPE_ENABLED,
    DEFAULT_HISTORY_SOURCE,
    HISTORY_BACKEND_JSONL,
    HISTORY_BACKEND_SQLITE,
    HISTORY_PREVIEW_CHARS,
    JSONL_SUFFIX,
    SQLITE_SUFFIXES,
)
from .sqlite_backend import SQLiteBackend


def resolve_history_backend(
    backend: str | None = None,
    history_path: str | None = None,
) -> str:
    """确定本次 history 操作使用哪个后端。

    优先级：
    1. 调用方显式传入 backend
    2. 根据请求级 history_path 后缀推断，比如 .sqlite3 / .jsonl
    3. 使用全局配置 CONVERSATION_HISTORY_BACKEND

    这样 eval 可以通过请求级 path 临时切到隔离 SQLite 文件，
    主服务也可以继续使用默认配置。
    """

    if backend:
        return backend.lower()

    if history_path:
        suffix = Path(history_path).suffix.lower()
        if suffix == JSONL_SUFFIX:
            return HISTORY_BACKEND_JSONL
        if suffix in SQLITE_SUFFIXES:
            return HISTORY_BACKEND_SQLITE

    configured_backend = CONVERSATION_HISTORY_CONFIG.backend.lower()
    if configured_backend in {HISTORY_BACKEND_SQLITE, HISTORY_BACKEND_JSONL}:
        return configured_backend
    return HISTORY_BACKEND_SQLITE


def resolve_history_path(
    backend: str | None = None,
    history_path: str | None = None,
) -> Path:
    """确定本次 history 操作读写哪个文件。

    history_path 是请求级覆盖，主要用于 eval 隔离；
    没有覆盖时再按 backend 选择默认 sqlite_path / jsonl_path。
    """

    if history_path:
        return Path(history_path)

    resolved_backend = resolve_history_backend(backend)
    if resolved_backend == HISTORY_BACKEND_JSONL:
        return Path(CONVERSATION_HISTORY_CONFIG.jsonl_path)
    return Path(CONVERSATION_HISTORY_CONFIG.sqlite_path)


def _get_backend(backend_name: str) -> HistoryBackend:
    """根据名字构造 backend 实例。

    这里每次调用都 new 一个实例，是为了让 CONVERSATION_HISTORY_CONFIG 在
    测试/eval 中被 monkeypatch 成新窗口时能立即生效。backend 本身无状态
    （不持有连接），创建成本可以忽略。
    """

    window = CONVERSATION_HISTORY_CONFIG.dedupe_window_seconds
    if backend_name == HISTORY_BACKEND_JSONL:
        return JsonlBackend(dedupe_window_seconds=window)
    return SQLiteBackend(dedupe_window_seconds=window)


# ---------- public high-level API ----------


def append_history_event(
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
    history_path: str | None = None,
    backend: str | None = None,
    dedupe: bool = DEFAULT_DEDUPE_ENABLED,
) -> dict[str, Any]:
    """追加一条非向量化会话流水。

    这里保存的是“发生过什么”，不是“用于语义召回的知识”。
    因此它适合支撑总结/回放，不适合直接参与 RAG 检索排序。
    """

    resolved_backend = resolve_history_backend(backend, history_path)
    path = resolve_history_path(resolved_backend, history_path)
    event = build_history_event(
        session_id=session_id,
        user_message=user_message,
        answer=answer,
        rewritten_query=rewritten_query,
        routes=routes,
        tags=tags,
        stored_to_vector=stored_to_vector,
        skipped_vector_store=skipped_vector_store,
        vector_store_skip_reason=vector_store_skip_reason,
        source=source,
    )
    return _get_backend(resolved_backend).append(event, path, dedupe)


def read_history_events(
    history_path: str | None = None,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    """读取指定后端中的全部合法会话流水。

    注意：这是治理工具接口，不建议主链路高频调用。
    主链路 summary 读取应使用 get_recent_history / get_all_history。
    """

    resolved_backend = resolve_history_backend(backend, history_path)
    path = resolve_history_path(resolved_backend, history_path)
    return _get_backend(resolved_backend).read_all(path)


def write_history_events(
    events: list[dict[str, Any]],
    history_path: str | None = None,
    backend: str | None = None,
) -> None:
    """用给定事件列表重写 history。

    主要给清理/治理脚本使用。主链路仍然只做 append，避免额外复杂度。
    """

    resolved_backend = resolve_history_backend(backend, history_path)
    path = resolve_history_path(resolved_backend, history_path)
    _get_backend(resolved_backend).write_all(events, path)


def get_recent_history(
    session_id: str,
    limit: int | None = None,
    history_path: str | None = None,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    """按时间顺序返回当前 session 最近几条会话流水。"""

    effective_limit = limit or CONVERSATION_HISTORY_CONFIG.recent_limit
    resolved_backend = resolve_history_backend(backend, history_path)
    path = resolve_history_path(resolved_backend, history_path)
    return _get_backend(resolved_backend).read_session(
        session_id=session_id,
        limit=effective_limit,
        path=path,
    )


def get_all_history(
    session_id: str,
    limit: int | None = None,
    history_path: str | None = None,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    """按时间顺序返回当前 session 的历史流水，默认只取最近一段窗口。"""

    effective_limit = limit or CONVERSATION_HISTORY_CONFIG.all_limit
    resolved_backend = resolve_history_backend(backend, history_path)
    path = resolve_history_path(resolved_backend, history_path)
    return _get_backend(resolved_backend).read_session(
        session_id=session_id,
        limit=effective_limit,
        path=path,
    )


def preview_history_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 event 列表裁成只包含展示字段的 preview 结构。"""

    return [
        {
            "preview": preview(
                event.get("rewritten_query") or event.get("user_message", ""),
                HISTORY_PREVIEW_CHARS,
            ),
            "timestamp": event.get("timestamp"),
            "routes": event.get("routes", []),
            "tags": event.get("tags", []),
            "stored_to_vector": event.get("stored_to_vector", False),
            "vector_store_skip_reason": event.get("vector_store_skip_reason", ""),
        }
        for event in events
    ]

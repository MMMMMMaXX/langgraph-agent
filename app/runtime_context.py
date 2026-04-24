from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

StreamCallback = Callable[[str, dict[str, Any]], None]


# stream_callback 是请求运行态对象，不属于可持久化 graph state。
# 使用 ContextVar 可以让同一进程内并发请求各自拿到自己的 SSE 回调，
# 同时避免 callable 被官方 SqliteSaver 写入 checkpoint。
_stream_callback: ContextVar[StreamCallback | None] = ContextVar(
    "stream_callback",
    default=None,
)


def set_stream_callback(callback: StreamCallback | None):
    """设置当前请求的流式回调，返回 token 供 finally 中恢复上下文。"""

    return _stream_callback.set(callback)


def reset_stream_callback(token) -> None:
    """恢复进入请求前的流式回调上下文。"""

    _stream_callback.reset(token)


def get_stream_callback() -> StreamCallback | None:
    """读取当前请求的流式回调。"""

    return _stream_callback.get()

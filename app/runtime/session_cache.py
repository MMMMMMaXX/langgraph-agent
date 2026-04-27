"""进程内 session cache 门面。

这一层基于 runtime 自己的 session backend，只做两件事：
1. 给 runtime 层提供统一 cache 读写接口
2. 把直接操作全局 dict 的细节关在一个文件里

后续如果要把内存 cache 换成别的实现，可以优先改这里，而不是让业务层到处改。
"""

from __future__ import annotations

import threading
from typing import Any

from app.runtime.session_backend import (
    get_backend_session_lock,
    session_store,
    session_store_guard,
)


def get_session_state(session_id: str) -> dict[str, Any] | None:
    """读取 session cache 中的状态副本。

    这里返回防御性拷贝，避免调用方无意中原地修改全局 session_store。
    """

    with session_store_guard:
        state = session_store.get(session_id)
        if state is None:
            return None
        return dict(state)


def set_session_state(session_id: str, state: dict[str, Any]) -> None:
    """写入 session cache。

    这里也做一次 dict 拷贝，避免上层后续继续修改入参对象时污染 cache。
    """

    with session_store_guard:
        session_store[session_id] = dict(state)


def clear_session_state(session_id: str) -> None:
    """删除单个 session 的内存缓存。"""

    with session_store_guard:
        session_store.pop(session_id, None)


def get_session_lock(session_id: str) -> threading.RLock:
    """透传单个 session 的互斥锁。"""

    return get_backend_session_lock(session_id)

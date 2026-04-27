"""进程内 session backend。

这里放真正的模块级共享状态与并发原语，是当前单机部署下 session 热缓存的底层实现。

之所以单独拆出来，是为了：
1. 让 runtime 层直接依赖自己的 backend，而不是反向 import API 兼容模块
2. 让 app/api/session_store.py 退化为向后兼容门面
"""

from __future__ import annotations

import threading

from app.state import AgentState

# 模块级共享状态（单机部署）；如果后续要做多进程/多机部署，整块需要换成外部存储。
session_store: dict[str, AgentState] = {}

# 保护 session_store 和 session_locks 两张字典“本身”的目录锁。
session_store_guard = threading.Lock()

# 每个 session 一把 RLock，保证同一 session 的请求串行，跨 session 仍可并发。
session_locks: dict[str, threading.RLock] = {}


def get_backend_session_lock(session_id: str) -> threading.RLock:
    """获取单个 session 的互斥锁（瞬时操作，不会阻塞业务）。"""

    with session_store_guard:
        lock = session_locks.get(session_id)
        if lock is None:
            lock = threading.RLock()
            session_locks[session_id] = lock
        return lock


def clear_in_memory_sessions() -> None:
    """只清空进程内 session cache 与锁表，不处理 checkpoint。"""

    with session_store_guard:
        session_store.clear()
        session_locks.clear()

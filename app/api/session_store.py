"""API 兼容门面：暴露 session 内存态给测试 / 脚本 / 老调用方。

真正的 session backend / 并发实现已经迁到 runtime 层：
- `app.runtime.session_backend`
- `app.runtime.session_cache`

这里保留的原因只有两个：
1. 维持 `from app.api import clear_session_store, session_store` 这类老用法
2. 让测试/脚本还能做 session 内省，而不需要知道 runtime 内部细节
"""

from __future__ import annotations

from app.checkpointing.factory import clear_checkpoints
from app.runtime.initial_state import create_initial_state
from app.runtime.session_backend import (
    clear_in_memory_sessions,
    session_store,
    session_store_guard,
)
from app.state import AgentState


def clear_session_store() -> None:
    """清空内存 session 状态和 LangGraph checkpoint，主要供 eval / 测试使用。"""

    clear_in_memory_sessions()
    clear_checkpoints()


def get_or_create_session_state(session_id: str) -> AgentState:
    """供外部直接读取 session 状态（测试/脚本）。

    注意：返回的是 session_store 内的引用，调用方不应修改。
    业务路径请用 `snapshot_session_state` 拿副本。
    """

    normalized_session_id = session_id.strip()
    if not normalized_session_id:
        raise ValueError("session_id must not be empty")

    with session_store_guard:
        if normalized_session_id not in session_store:
            session_store[normalized_session_id] = create_initial_state(
                normalized_session_id
            )
        return session_store[normalized_session_id]

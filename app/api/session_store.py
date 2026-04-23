"""Session 内存存储 + 并发模型。

单机多线程场景下，我们用两层锁保证 session 粒度的串行 + 全局字典安全：

1. session_store
   全局 session 状态字典（内存）。只存可持久字段：session_id / messages / summary。
2. session_store_guard
   保护 session_store 和 session_locks 两张字典“本身”的目录锁。
   只用于瞬时 get/put，不包含任何业务或长时间阻塞操作。
3. session_locks
   每个 session 一把 RLock，保证同一 session 的请求串行，
   跨 session 请求仍可并发执行 graph.invoke。

锁顺序约定（严格遵守，任何新增调用必须遵循）：

  (A) 先取 session_store_guard（短暂）查/建 session_lock，立即释放
  (B) 再取 session_lock 执行 graph.invoke（长时间）
  (C) 需要读写 session_store 时，在 session_lock 内再短暂取 guard

反向嵌套（先 session_lock 再 guard 长期持有）会造成死锁，禁止。

chat_runner.build_chat_result 是唯一需要遵循全部顺序的调用方。
外部（测试/脚本）读 session 请用 `get_or_create_session_state`，
它只短暂持有 guard、不涉及 session_lock。
"""

from __future__ import annotations

import threading
from typing import Any

from app.checkpointing.factory import clear_checkpoints
from app.chat_service import create_initial_state
from app.state import AgentState

# 模块级共享状态（单机部署）；如果后续要做多进程/多机部署，整块需要换成外部存储。
session_store: dict[str, AgentState] = {}
session_store_guard = threading.Lock()
session_locks: dict[str, threading.RLock] = {}


def get_session_lock(session_id: str) -> threading.RLock:
    """获取单个 session 的互斥锁（瞬时操作，不会阻塞业务）。

    只在 guard 内做字典查/建，立刻释放 guard，不持有任何长时间锁。
    """

    with session_store_guard:
        lock = session_locks.get(session_id)
        if lock is None:
            lock = threading.RLock()
            session_locks[session_id] = lock
        return lock


def clear_session_store() -> None:
    """清空内存 session 状态和 LangGraph checkpoint，主要供 eval / 测试使用。"""

    with session_store_guard:
        session_store.clear()
        session_locks.clear()
    clear_checkpoints()


def snapshot_session_state(session_id: str) -> AgentState:
    """在 guard 内拿到 session 状态的浅拷贝快照后立即释放 guard。

    返回的 dict 与 session_store 里的对象是不同引用，后续读写不会相互影响。
    供 chat_runner 持有 session_lock 期间使用。
    """

    with session_store_guard:
        state = session_store.get(session_id)
        if state is None:
            state = create_initial_state(session_id)
            session_store[session_id] = state
        # 浅拷贝足够：后续 run_chat_turn 会基于这个 dict 构造新对象，
        # 不会原地修改 messages / summary 等字段。
        return dict(state)


def _result_to_persistent_state(
    session_id: str, result: dict[str, Any]
) -> AgentState:
    """把 run_chat_turn 的返回值裁成“只保留需要持久化的字段”的状态。

    把字段列表集中在这里，未来新增需要跨 turn 保留的字段（例如 user_profile）
    只改这一个函数，不用在 API 层到处找写回点。
    """

    return {
        "session_id": session_id,
        "messages": result.get("messages", []),
        "summary": result.get("summary", ""),
    }


def commit_session_state(session_id: str, result: dict[str, Any]) -> None:
    """把 run_chat_turn 的结果写回 session_store（瞬时 guard）。"""

    persistent = _result_to_persistent_state(session_id, result)
    with session_store_guard:
        session_store[session_id] = persistent


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

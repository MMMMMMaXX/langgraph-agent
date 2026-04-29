"""会话运行时门面。

这一层统一管理：
1. 从哪里恢复会话状态（cache / checkpoint / empty）
2. 如何在 turn 完成后提交最终状态（checkpoint + cache）

当前版本刻意保持“语义不变”，只做收口，不改变原有 session/cache/checkpoint 的职责。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from app.constants.runtime import (
    RUNTIME_RESTORE_FROM_CACHE,
)
from app.runtime.checkpoint_store import (
    load_checkpoint_snapshot,
    persist_final_checkpoint_state,
)
from app.runtime.initial_state import create_initial_state
from app.runtime.session_cache import (
    clear_session_state,
    get_session_lock,
    get_session_state,
    set_session_state,
)
from app.runtime.snapshot import ConversationSnapshot
from app.state import AgentState

StreamCallback = Callable[[str, dict[str, Any]], None]


class SessionRuntime:
    """统一管理单个 session 的加载、提交与清理。"""

    def get_lock(self, session_id: str) -> threading.RLock:
        """返回单个 session 的互斥锁。

        API 层仍负责决定“在哪里持锁、持锁多久”，但不需要再直接感知
        session_store / session_locks 这些底层实现细节。
        """

        return get_session_lock(session_id)

    def build_request_state(
        self,
        *,
        session_id: str,
        request_id: str,
        debug: bool,
        conversation_history_path: str,
        stream_callback: StreamCallback | None,
    ) -> AgentState:
        """构造一次 turn 的请求态。

        这一步只从进程内 cache 拿“热状态”，不触发 checkpoint 恢复。
        冷恢复仍由 chat_service 在真正执行 graph 前处理，避免 API 层同时知道
        cache 和 checkpoint 两套状态源。
        """

        cached_state = get_session_state(session_id)
        if cached_state is None:
            cached_state = create_initial_state(session_id)
            # 保持与旧 snapshot_session_state 一致：首次访问 session 时就让
            # 内存 cache 中存在一个最小初始态，方便后续测试/内省路径保持兼容。
            set_session_state(session_id, cached_state)

        return {
            **cached_state,
            "request_id": request_id,
            "session_id": session_id,
            "debug": debug,
            "conversation_history_path": conversation_history_path,
            "stream_callback": stream_callback,
            "streamed_answer": False,
        }

    def cache_turn_result(
        self, session_id: str, result: dict[str, Any]
    ) -> AgentState:
        """把本轮结果裁成精简状态后写回进程内 cache。

        这里故意只保留跨 turn 真正需要继续带下去的字段，避免把 request 级噪声
        （如 request_id / debug_info / node_timings）塞进长期会话态。
        """

        persistent_state: AgentState = {
            "session_id": session_id,
            "messages": result.get("messages", []),
            "summary": result.get("summary", ""),
        }
        set_session_state(session_id, persistent_state)
        return persistent_state

    def load(self, session_id: str, graph: Any) -> ConversationSnapshot:
        """加载当前 session 的会话快照。

        这里有一个很容易踩坑的边界：
        chat_runner 在新 session 进入时，会先把一个“空初始 state”放进内存 cache。
        如果我们简单地把“cache 里有 dict”当成命中，就会错过 checkpoint 恢复。

        所以这里的规则是：
        - cache 里必须真的有 messages 或 summary，才算有效命中
        - 否则继续回退到 checkpoint
        """

        cached_state = get_session_state(session_id)
        cached_messages = list((cached_state or {}).get("messages", []))
        cached_summary = (cached_state or {}).get("summary", "")

        if cached_messages or cached_summary:
            return ConversationSnapshot(
                session_id=session_id,
                thread_id=session_id,
                messages=cached_messages,
                summary=cached_summary,
                restored_from=RUNTIME_RESTORE_FROM_CACHE,
                has_checkpoint_state=False,
            )

        return load_checkpoint_snapshot(graph, session_id)

    def commit(
        self,
        session_id: str,
        graph: Any,
        state: dict[str, Any],
        answer: str,
    ) -> list[dict[str, Any]]:
        """提交本轮最终状态。

        提交流程分两步：
        1. 先把最终 assistant 消息补写回 checkpoint，保证重启恢复不丢最后一轮
        2. 再把精简后的可持久字段写回进程内 cache，供同进程后续请求快速复用
        """

        updated_messages = persist_final_checkpoint_state(
            graph=graph,
            session_id=session_id,
            state=state,
            answer=answer,
        )
        set_session_state(
            session_id,
            {
                "session_id": session_id,
                "messages": updated_messages,
                "summary": state.get("summary", ""),
            },
        )
        return updated_messages

    def clear(self, session_id: str) -> None:
        """清理单个 session 的内存缓存。

        当前版本只负责清 cache；checkpoint 的清理仍沿用现有测试/eval 清理入口。
        """

        clear_session_state(session_id)

"""SessionRuntime 单测。

这一组测试只验证 runtime 层自己的核心契约：
1. cache 热命中是否优先返回
2. 空初始 state 是否会继续回退到 checkpoint
3. commit 是否会同时更新 checkpoint 与内存 cache
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from app.api import clear_session_store
from app.constants.runtime import (
    RUNTIME_RESTORE_FROM_CACHE,
    RUNTIME_RESTORE_FROM_CHECKPOINT,
)
from app.runtime.initial_state import create_initial_state
from app.runtime.session_cache import get_session_state, set_session_state
from app.runtime.session_runtime import SessionRuntime


@pytest.fixture(autouse=True)
def _reset_sessions() -> Iterator[None]:
    clear_session_store()
    yield
    clear_session_store()


class _FakeGraph:
    """最小 graph 桩：只实现 runtime 当前需要的两个方法。"""

    def __init__(self, checkpoint_values: dict[str, Any] | None = None) -> None:
        self.checkpoint_values = checkpoint_values or {}
        self.get_state_calls: list[dict[str, Any]] = []
        self.update_state_calls: list[dict[str, Any]] = []

    def get_state(self, config: dict[str, Any]) -> Any:
        self.get_state_calls.append(config)
        return SimpleNamespace(values=self.checkpoint_values)

    def update_state(
        self,
        config: dict[str, Any],
        payload: dict[str, Any],
        *,
        as_node: str,
    ) -> None:
        self.update_state_calls.append(
            {"config": config, "payload": payload, "as_node": as_node}
        )


def test_load_prefers_non_empty_cache() -> None:
    runtime = SessionRuntime()
    graph = _FakeGraph(
        checkpoint_values={
            "messages": [{"role": "user", "content": "from-checkpoint"}],
            "summary": "checkpoint-summary",
        }
    )
    set_session_state(
        "u1",
        {
            "session_id": "u1",
            "messages": [{"role": "user", "content": "from-cache"}],
            "summary": "cache-summary",
        },
    )

    snapshot = runtime.load("u1", graph)

    assert snapshot.restored_from == RUNTIME_RESTORE_FROM_CACHE
    assert snapshot.messages[0]["content"] == "from-cache"
    assert snapshot.summary == "cache-summary"
    assert graph.get_state_calls == []


def test_load_ignores_empty_initial_cache_and_falls_back_to_checkpoint() -> None:
    runtime = SessionRuntime()
    graph = _FakeGraph(
        checkpoint_values={
            "messages": [{"role": "user", "content": "from-checkpoint"}],
            "summary": "checkpoint-summary",
        }
    )
    # 这就是 chat_runner 对新 session 常见的初始空 state 形态。
    set_session_state("u1", create_initial_state("u1"))

    snapshot = runtime.load("u1", graph)

    assert snapshot.restored_from == RUNTIME_RESTORE_FROM_CHECKPOINT
    assert snapshot.messages[0]["content"] == "from-checkpoint"
    assert snapshot.summary == "checkpoint-summary"
    assert len(graph.get_state_calls) == 1


def test_commit_updates_checkpoint_and_cache() -> None:
    runtime = SessionRuntime()
    graph = _FakeGraph()

    updated_messages = runtime.commit(
        session_id="u1",
        graph=graph,
        state={
            "messages": [{"role": "user", "content": "hello"}],
            "summary": "summary-v1",
        },
        answer="world",
    )

    assert updated_messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    assert len(graph.update_state_calls) == 1
    call = graph.update_state_calls[0]
    assert call["config"] == {"configurable": {"thread_id": "u1"}}
    assert call["payload"]["summary"] == "summary-v1"
    assert call["payload"]["messages"] == updated_messages

    cached = get_session_state("u1")
    assert cached is not None
    assert cached["session_id"] == "u1"
    assert cached["messages"] == updated_messages
    assert cached["summary"] == "summary-v1"


def test_build_request_state_initializes_empty_cache_on_first_access() -> None:
    runtime = SessionRuntime()

    state = runtime.build_request_state(
        session_id="u1",
        request_id="req-1",
        debug=True,
        conversation_history_path="data/history.sqlite3",
        stream_callback=None,
    )

    assert state["session_id"] == "u1"
    assert state["request_id"] == "req-1"
    assert state["debug"] is True
    assert state["conversation_history_path"] == "data/history.sqlite3"
    assert state["messages"] == []
    assert state["summary"] == ""

    cached = get_session_state("u1")
    assert cached is not None
    assert cached["session_id"] == "u1"
    assert cached["messages"] == []
    assert cached["summary"] == ""


def test_cache_turn_result_only_persists_cross_turn_fields() -> None:
    runtime = SessionRuntime()

    persistent = runtime.cache_turn_result(
        "u1",
        {
            "session_id": "u1",
            "messages": [{"role": "user", "content": "hi"}],
            "summary": "summary-v1",
            "request_id": "req-1",
            "debug_info": {"chat_agent": {"foo": "bar"}},
            "node_timings": {"chat_agent": 1.23},
        },
    )

    assert persistent == {
        "session_id": "u1",
        "messages": [{"role": "user", "content": "hi"}],
        "summary": "summary-v1",
    }
    cached = get_session_state("u1")
    assert cached == persistent

"""Unit tests: app/agents/workflow_executor.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from app.agents.workflow_executor import (
    run_chat_step,
    run_rag_step,
    run_tool_step,
    workflow_executor_node,
)
from app.auth.context import AuthContext
from app.constants.auth import (
    ANONYMOUS_TENANT_ID,
    ANONYMOUS_USER_ID,
    ROLE_ANONYMOUS,
)
from app.constants.tool_safety import CONFIRMATION_SECRET_ENV
from app.constants.tooling import (
    TOOL_NAME_GET_WEATHER,
    TOOL_NAME_TICKET_CREATE,
)
from app.constants.workflow import (
    NODE_WORKFLOW_EXECUTOR,
    STEP_STATUS_FAILED,
    STEP_STATUS_NEED_CONFIRMATION,
    STEP_STATUS_SKIPPED,
    STEP_STATUS_SUCCEEDED,
    WORKFLOW_STATUS_FAILED,
    WORKFLOW_STATUS_NEED_CONFIRMATION,
    WORKFLOW_STATUS_SUCCEEDED,
)
from app.tools import execution_record as exec_record_mod
from app.tools import ticket as ticket_mod
from app.tools.pipeline import SideEffectContext
from app.workflow.registry import default_tool_registry

# ---------------------------- fixtures ----------------------------


@dataclass(frozen=True)
class _TmpOpsConfig:
    path: str


@pytest.fixture(autouse=True)
def _secret(monkeypatch: pytest.MonkeyPatch) -> None:
    # side_effect step 会调用 issue_token / verify_token，需要 secret。
    monkeypatch.setenv(CONFIRMATION_SECRET_ENV, "workflow-test-secret")


@pytest.fixture()
def ops_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """把 tool_executions / mock_tickets 的存储指向临时目录，测试隔离。"""

    db = tmp_path / "operations.sqlite3"
    cfg = _TmpOpsConfig(path=str(db))
    monkeypatch.setattr(exec_record_mod, "OPERATIONS_CONFIG", cfg)
    monkeypatch.setattr(ticket_mod, "OPERATIONS_CONFIG", cfg)
    return db


def _auth(anonymous: bool = False) -> AuthContext:
    if anonymous:
        return AuthContext(
            tenant_id=ANONYMOUS_TENANT_ID,
            user_id=ANONYMOUS_USER_ID,
            role=ROLE_ANONYMOUS,
            anonymous=True,
        )
    return AuthContext(tenant_id="t1", user_id="u1", role="user", anonymous=False)


def _state(
    *,
    plan: dict | None = None,
    anonymous: bool = False,
    confirmation_token: str = "",
) -> dict:
    return {
        "messages": [{"role": "user", "content": "workflow test message"}],
        "auth": _auth(anonymous=anonymous),
        "request_id": "req-wf",
        "session_id": "sess-wf",
        "plan": plan or {},
        "plan_id": "plan-fixed",
        "confirmation_token": confirmation_token,
    }


def _side_ctx(auth: AuthContext | None = None) -> SideEffectContext:
    return SideEffectContext(
        auth=auth or _auth(),
        session_id="sess-wf",
        request_id="req-wf",
        confirmation_token="",
    )


# ---------------------------- run_tool_step ----------------------------


def test_run_tool_step_read_only_happy_path() -> None:
    step = {
        "id": "s1",
        "agent": "tool_agent",
        "purpose": "weather",
        "tool": "get_weather",
        "args": {"city": "北京"},
    }
    result = run_tool_step(
        step, _state(), registry=default_tool_registry, side_ctx=_side_ctx()
    )
    assert result["status"] == STEP_STATUS_SUCCEEDED
    assert result["tool_name"] == TOOL_NAME_GET_WEATHER
    assert "北京" in result["output"]


def test_run_tool_step_drops_planner_hallucinated_args() -> None:
    # Planner 经常在 free-form JSON 里塞 spec 未声明的字段（如给 get_weather
    # 塞 `location`）。真实 eval 观察到过 TypeError。Executor 应按白名单过滤
    # 未知参数，保证工具能成功调到。
    step = {
        "id": "s1",
        "agent": "tool_agent",
        "purpose": "weather",
        "tool": "get_weather",
        "args": {"city": "北京", "location": "city-level", "level": "info"},
    }
    result = run_tool_step(
        step, _state(), registry=default_tool_registry, side_ctx=_side_ctx()
    )
    assert result["status"] == STEP_STATUS_SUCCEEDED
    # 保留审计里真正用到的 args；未知 key 被丢掉，不在 result["args"] 里。
    assert result["args"] == {"city": "北京"}
    assert "北京" in result["output"]


def test_run_tool_step_resolves_business_name() -> None:
    # Planner 常以"业务名"书写；registry.resolve 会把点号映射为下划线。
    # get_weather 没有点号版本，这里用 calculate 的别名形式验证。
    step = {
        "id": "s1",
        "agent": "tool_agent",
        "purpose": "calc",
        "tool": "calculate",
        "args": {"expression": "1+2"},
    }
    result = run_tool_step(
        step, _state(), registry=default_tool_registry, side_ctx=_side_ctx()
    )
    assert result["status"] == STEP_STATUS_SUCCEEDED
    assert "3" in result["output"]


def test_run_tool_step_unknown_tool_fails() -> None:
    step = {
        "id": "s1",
        "agent": "tool_agent",
        "purpose": "x",
        "tool": "ghost.tool",
        "args": {},
    }
    result = run_tool_step(
        step, _state(), registry=default_tool_registry, side_ctx=_side_ctx()
    )
    assert result["status"] == STEP_STATUS_FAILED
    assert result["error"] == "tool_not_resolvable"


def test_run_tool_step_anonymous_blocks_side_effect() -> None:
    step = {
        "id": "s1",
        "agent": "tool_agent",
        "purpose": "create ticket",
        "tool": "ticket.create",
        "args": {"title": "bug"},
    }
    result = run_tool_step(
        step,
        _state(anonymous=True),
        registry=default_tool_registry,
        side_ctx=_side_ctx(auth=_auth(anonymous=True)),
    )
    assert result["status"] == STEP_STATUS_FAILED


def test_run_tool_step_side_effect_returns_need_confirmation(ops_db: Path) -> None:
    step = {
        "id": "s1",
        "agent": "tool_agent",
        "purpose": "create ticket",
        "tool": "ticket.create",
        "args": {"title": "bug"},
    }
    ctx = _side_ctx()
    result = run_tool_step(step, _state(), registry=default_tool_registry, side_ctx=ctx)
    assert result["status"] == STEP_STATUS_NEED_CONFIRMATION
    pc = result["pending_confirmation"]
    assert pc["tool_name"] == TOOL_NAME_TICKET_CREATE
    assert pc["args"] == {"title": "bug"}
    assert pc["token"]
    # need_confirmation 触发后，ctx.pending_confirmation 应被执行器清空，
    # 避免下一个 step 误用同一份 pending 标记。
    assert ctx.pending_confirmation is None


# ---------------------------- run_rag_step ----------------------------


def test_run_rag_step_delegates_to_rag_agent(monkeypatch) -> None:
    # 用 monkeypatch 把 rag_agent_node 换成 deterministic stub，避免触发真实
    # embedding + 向量检索。executor 负责合成 state 并读 answer 即可。
    from app.agents import rag_agent as rag_mod

    def fake_rag(synth_state: dict) -> dict:
        # 验证 executor 把 step.query 作为最新 message 传进来。
        assert synth_state["messages"][-1]["content"] == "什么是 WAI-ARIA"
        return {"answer": "WAI-ARIA 是可访问性规范。"}

    monkeypatch.setattr(rag_mod, "rag_agent_node", fake_rag)

    step = {
        "id": "s1",
        "agent": "rag_agent",
        "purpose": "retrieve",
        "query": "什么是 WAI-ARIA",
    }
    result = run_rag_step(step, _state())
    assert result["status"] == STEP_STATUS_SUCCEEDED
    assert "WAI-ARIA" in result["output"]


# ---------------------------- run_chat_step ----------------------------


def test_run_chat_step_returns_purpose_as_output() -> None:
    step = {
        "id": "s1",
        "agent": "chat_agent",
        "purpose": "向用户说明结果",
    }
    result = run_chat_step(step, _state())
    assert result["status"] == STEP_STATUS_SUCCEEDED
    assert result["output"] == "向用户说明结果"


# ---------------------------- workflow_executor_node ----------------------------


def _plan_multi(steps: list[dict]) -> dict:
    return {
        "task_type": "multi",
        "steps": steps,
        "compose_goal": "",
    }


def test_executor_empty_plan_passes_through() -> None:
    """Planner fail-closed 时 plan 为空，Executor 不再写 verification，
    只记录一次 debug_info，保持 Planner 的 workflow_status=failed。"""

    state = _state(plan={})
    state["workflow_status"] = WORKFLOW_STATUS_FAILED
    result = workflow_executor_node(state)
    assert result["workflow_status"] == WORKFLOW_STATUS_FAILED
    assert result["step_results"] == {}
    assert result["debug_info"][NODE_WORKFLOW_EXECUTOR]["status"] == "noop"


def test_executor_runs_sequential_steps_happy_path() -> None:
    plan = _plan_multi(
        [
            {
                "id": "s1",
                "agent": "tool_agent",
                "purpose": "weather",
                "tool": "get_weather",
                "args": {"city": "北京"},
            },
            {
                "id": "s2",
                "agent": "chat_agent",
                "purpose": "告诉用户结果",
                "depends_on": ["s1"],
            },
        ]
    )
    result = workflow_executor_node(_state(plan=plan))
    assert result["workflow_status"] == WORKFLOW_STATUS_SUCCEEDED
    assert set(result["step_results"].keys()) == {"s1", "s2"}
    assert all(
        r["status"] == STEP_STATUS_SUCCEEDED for r in result["step_results"].values()
    )
    # fallback answer 至少拼进了第一步的输出
    answer = result["agent_outputs"][NODE_WORKFLOW_EXECUTOR]
    assert "北京" in answer


def test_executor_short_circuits_on_need_confirmation(ops_db: Path) -> None:
    plan = _plan_multi(
        [
            {
                "id": "s1",
                "agent": "tool_agent",
                "purpose": "create ticket",
                "tool": "ticket.create",
                "args": {"title": "bug"},
            },
            {
                "id": "s2",
                "agent": "chat_agent",
                "purpose": "告诉用户",
                "depends_on": ["s1"],
            },
        ]
    )
    result = workflow_executor_node(_state(plan=plan))
    assert result["workflow_status"] == WORKFLOW_STATUS_NEED_CONFIRMATION
    assert result["step_results"]["s1"]["status"] == STEP_STATUS_NEED_CONFIRMATION
    # s2 未执行，被标记为 skipped（后续节点需要看见清晰的状态）
    assert result["step_results"]["s2"]["status"] == STEP_STATUS_SKIPPED
    # pending_confirmation 透传到 state，merge / API 层据此提示客户端
    assert result["pending_confirmation"]["tool_name"] == TOOL_NAME_TICKET_CREATE


def test_executor_short_circuits_on_failure() -> None:
    plan = _plan_multi(
        [
            {
                "id": "s1",
                "agent": "tool_agent",
                "purpose": "bad",
                "tool": "ghost.tool",
                "args": {},
            },
            {
                "id": "s2",
                "agent": "chat_agent",
                "purpose": "x",
                "depends_on": ["s1"],
            },
        ]
    )
    result = workflow_executor_node(_state(plan=plan))
    assert result["workflow_status"] == WORKFLOW_STATUS_FAILED
    assert result["step_results"]["s1"]["status"] == STEP_STATUS_FAILED
    assert result["step_results"]["s2"]["status"] == STEP_STATUS_SKIPPED

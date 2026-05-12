"""Unit tests: app/agents/verifier_agent.py."""

from __future__ import annotations

import pytest

from app.agents.verifier_agent import verifier_node
from app.auth.context import AuthContext
from app.constants.auth import (
    ANONYMOUS_TENANT_ID,
    ANONYMOUS_USER_ID,
    ROLE_ANONYMOUS,
)
from app.constants.workflow import (
    ERR_PLAN_LLM_FAILED,
    ERR_VERIFY_MISSING_ARGS,
    ERR_VERIFY_STEP_FAILED,
    ERR_VERIFY_TOOL_UNAUTHORIZED,
    NODE_VERIFIER,
    RISK_WARN_HIGH_RISK_TOOL,
    RISK_WARN_SIDE_EFFECT_CONFIRMED,
    STEP_STATUS_FAILED,
    STEP_STATUS_NEED_CONFIRMATION,
    STEP_STATUS_SUCCEEDED,
    VERIFICATION_STATUS_FAILED,
    VERIFICATION_STATUS_NEED_CLARIFICATION,
    VERIFICATION_STATUS_NEED_CONFIRMATION,
    VERIFICATION_STATUS_PASS,
    WORKFLOW_STATUS_FAILED,
    WORKFLOW_STATUS_NEED_CLARIFICATION,
    WORKFLOW_STATUS_NEED_CONFIRMATION,
    WORKFLOW_STATUS_SUCCEEDED,
)


# ---------------------------- helpers ----------------------------


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
    step_results: dict | None = None,
    workflow_status: str = WORKFLOW_STATUS_SUCCEEDED,
    verification: dict | None = None,
    anonymous: bool = False,
) -> dict:
    return {
        "messages": [{"role": "user", "content": "verifier test"}],
        "auth": _auth(anonymous=anonymous),
        "request_id": "req-v",
        "session_id": "sess-v",
        "plan": plan or {},
        "plan_id": "plan-v",
        "step_results": step_results or {},
        "workflow_status": workflow_status,
        "verification": verification or {},
    }


def _plan(steps: list[dict]) -> dict:
    return {"task_type": "multi", "steps": steps, "compose_goal": ""}


def _tool_step(
    step_id: str, tool: str, args: dict | None = None
) -> dict:
    return {
        "id": step_id,
        "agent": "tool_agent",
        "purpose": "p",
        "tool": tool,
        "args": args or {},
        "query": None,
        "depends_on": [],
    }


# ---------------------------- empty plan (planner fail-closed) ----------------------------


def test_verifier_passes_through_planner_failure() -> None:
    existing = {
        "status": VERIFICATION_STATUS_FAILED,
        "missing_fields": [],
        "unsupported_claims": [ERR_PLAN_LLM_FAILED],
        "risk_warnings": [],
    }
    state = _state(
        plan={},
        verification=existing,
        workflow_status=WORKFLOW_STATUS_FAILED,
    )
    out = verifier_node(state)
    assert out["verification"] == existing
    assert out["workflow_status"] == WORKFLOW_STATUS_FAILED
    assert out["debug_info"][NODE_VERIFIER]["status"] == "noop"


def test_verifier_fills_missing_verification_fields_on_empty_plan() -> None:
    # Planner 理论上永远会写 verification，但防御性测试：即便 plan 为空
    # 且没有 verification，verifier 也应返回结构完整的字段。
    state = _state(plan={}, workflow_status=WORKFLOW_STATUS_FAILED)
    state["verification"] = {}
    out = verifier_node(state)
    v = out["verification"]
    assert v["status"] == VERIFICATION_STATUS_FAILED
    assert v["missing_fields"] == []
    assert v["unsupported_claims"] == []
    assert v["risk_warnings"] == []


# ---------------------------- happy path ----------------------------


def test_verifier_passes_clean_read_only_plan() -> None:
    plan = _plan([_tool_step("s1", "get_weather", {"city": "北京"})])
    step_results = {"s1": {"status": STEP_STATUS_SUCCEEDED, "output": "晴"}}
    out = verifier_node(_state(plan=plan, step_results=step_results))

    v = out["verification"]
    assert v["status"] == VERIFICATION_STATUS_PASS
    assert v["missing_fields"] == []
    assert v["unsupported_claims"] == []
    # get_weather 是 low-risk read_only，没有任何风险提示
    assert v["risk_warnings"] == []
    assert out["workflow_status"] == WORKFLOW_STATUS_SUCCEEDED


# ---------------------------- missing args ----------------------------


def test_verifier_reports_missing_required_args() -> None:
    plan = _plan([_tool_step("s1", "get_weather", args={})])
    step_results = {"s1": {"status": STEP_STATUS_SUCCEEDED, "output": ""}}
    out = verifier_node(_state(plan=plan, step_results=step_results))

    v = out["verification"]
    assert "s1.city" in v["missing_fields"]
    # 执行层 succeeded + 缺参 → need_clarification
    assert v["status"] == VERIFICATION_STATUS_NEED_CLARIFICATION
    assert out["workflow_status"] == WORKFLOW_STATUS_NEED_CLARIFICATION


def test_verifier_treats_empty_string_as_missing() -> None:
    plan = _plan([_tool_step("s1", "get_weather", args={"city": ""})])
    step_results = {"s1": {"status": STEP_STATUS_SUCCEEDED, "output": ""}}
    out = verifier_node(_state(plan=plan, step_results=step_results))
    assert "s1.city" in out["verification"]["missing_fields"]


# ---------------------------- unauthorized ----------------------------


def test_verifier_flags_anonymous_side_effect_as_unauthorized() -> None:
    plan = _plan([_tool_step("s1", "ticket.create", {"title": "bug"})])
    # 执行器匿名会把它标 failed；verifier 仍然把越权原因写进 claims。
    step_results = {
        "s1": {"status": STEP_STATUS_FAILED, "output": "", "error": "tool_not_resolvable"}
    }
    out = verifier_node(
        _state(
            plan=plan,
            step_results=step_results,
            workflow_status=WORKFLOW_STATUS_FAILED,
            anonymous=True,
        )
    )
    v = out["verification"]
    assert f"{ERR_VERIFY_TOOL_UNAUTHORIZED}:s1" in v["unsupported_claims"]
    # 越权挂 side_effect 风险提示
    assert RISK_WARN_SIDE_EFFECT_CONFIRMED in v["risk_warnings"]
    assert v["status"] == VERIFICATION_STATUS_FAILED
    assert out["workflow_status"] == WORKFLOW_STATUS_FAILED


# ---------------------------- step failure -----------------------------


def test_verifier_translates_step_failure_into_claim() -> None:
    plan = _plan([_tool_step("s1", "get_weather", {"city": "北京"})])
    step_results = {"s1": {"status": STEP_STATUS_FAILED, "output": "", "error": "boom"}}
    out = verifier_node(
        _state(
            plan=plan,
            step_results=step_results,
            workflow_status=WORKFLOW_STATUS_FAILED,
        )
    )
    v = out["verification"]
    assert f"{ERR_VERIFY_STEP_FAILED}:s1" in v["unsupported_claims"]
    assert v["status"] == VERIFICATION_STATUS_FAILED
    assert out["workflow_status"] == WORKFLOW_STATUS_FAILED


# ---------------------------- need_confirmation ----------------------------


def test_verifier_preserves_need_confirmation_and_flags_side_effect() -> None:
    plan = _plan([_tool_step("s1", "ticket.create", {"title": "bug"})])
    step_results = {
        "s1": {
            "status": STEP_STATUS_NEED_CONFIRMATION,
            "output": "请确认",
            "pending_confirmation": {"token": "t", "tool_name": "ticket_create"},
        }
    }
    out = verifier_node(
        _state(
            plan=plan,
            step_results=step_results,
            workflow_status=WORKFLOW_STATUS_NEED_CONFIRMATION,
        )
    )
    v = out["verification"]
    assert v["status"] == VERIFICATION_STATUS_NEED_CONFIRMATION
    assert RISK_WARN_SIDE_EFFECT_CONFIRMED in v["risk_warnings"]
    # ticket.create 是 medium risk，要挂 high_risk_tool
    assert RISK_WARN_HIGH_RISK_TOOL in v["risk_warnings"]
    assert out["workflow_status"] == WORKFLOW_STATUS_NEED_CONFIRMATION


# ---------------------------- risk warnings dedup --------------------------


def test_verifier_dedups_risk_warnings_across_steps() -> None:
    plan = _plan(
        [
            _tool_step("s1", "ticket.create", {"title": "a"}),
            _tool_step("s2", "ticket.create", {"title": "b"}),
        ]
    )
    step_results = {
        "s1": {"status": STEP_STATUS_SUCCEEDED, "output": "#1"},
        "s2": {"status": STEP_STATUS_SUCCEEDED, "output": "#2"},
    }
    out = verifier_node(_state(plan=plan, step_results=step_results))
    risks = out["verification"]["risk_warnings"]
    # 相同 warning 只保留一次
    assert risks.count(RISK_WARN_SIDE_EFFECT_CONFIRMED) == 1
    assert risks.count(RISK_WARN_HIGH_RISK_TOOL) == 1


# ---------------------------- status merge priority ------------------------


def test_verifier_upgrades_succeeded_to_need_clarification_on_missing_args() -> None:
    # Executor 乐观汇报 succeeded，Verifier 发现缺参 → 合并后 need_clarification。
    plan = _plan([_tool_step("s1", "get_weather", args={})])
    step_results = {"s1": {"status": STEP_STATUS_SUCCEEDED, "output": ""}}
    out = verifier_node(_state(plan=plan, step_results=step_results))
    assert out["workflow_status"] == WORKFLOW_STATUS_NEED_CLARIFICATION


def test_verifier_keeps_failed_when_executor_already_failed() -> None:
    plan = _plan([_tool_step("s1", "get_weather", {"city": "北京"})])
    step_results = {"s1": {"status": STEP_STATUS_FAILED, "output": ""}}
    out = verifier_node(
        _state(
            plan=plan,
            step_results=step_results,
            workflow_status=WORKFLOW_STATUS_FAILED,
        )
    )
    # executor failed 是最严重档，无论 verifier 怎么推都得保持 failed
    assert out["workflow_status"] == WORKFLOW_STATUS_FAILED


# ---------------------------- unknown tool defence ------------------------


def test_verifier_flags_unknown_tool_as_step_failed() -> None:
    """Planner 未知工具时正常会 fail-closed；这里测试若某种原因它仍进来 verifier，
    verifier 会把该 step 记一条 step_failed claim，不抛。"""

    plan = _plan([_tool_step("s1", "ghost.tool", {})])
    step_results = {"s1": {"status": STEP_STATUS_FAILED, "output": ""}}
    out = verifier_node(
        _state(
            plan=plan,
            step_results=step_results,
            workflow_status=WORKFLOW_STATUS_FAILED,
        )
    )
    v = out["verification"]
    # 未知工具走 ToolNotRegisteredError 分支，会登记 step_failed claim
    assert any(c.startswith(ERR_VERIFY_STEP_FAILED) for c in v["unsupported_claims"])
    assert v["status"] == VERIFICATION_STATUS_FAILED


# ---------------------------- rag/chat steps ---------------------------


def test_verifier_ignores_rag_and_chat_steps_for_arg_rules() -> None:
    plan = _plan(
        [
            {
                "id": "s1",
                "agent": "rag_agent",
                "purpose": "retrieve",
                "tool": None,
                "args": {},
                "query": "什么是 WAI-ARIA",
                "depends_on": [],
            },
            {
                "id": "s2",
                "agent": "chat_agent",
                "purpose": "总结",
                "tool": None,
                "args": {},
                "query": None,
                "depends_on": ["s1"],
            },
        ]
    )
    step_results = {
        "s1": {"status": STEP_STATUS_SUCCEEDED, "output": "..."},
        "s2": {"status": STEP_STATUS_SUCCEEDED, "output": "..."},
    }
    out = verifier_node(_state(plan=plan, step_results=step_results))
    v = out["verification"]
    # rag/chat 不受 required_args 规则影响
    assert v["missing_fields"] == []
    assert v["status"] == VERIFICATION_STATUS_PASS

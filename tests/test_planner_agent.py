"""Unit tests: app/agents/planner_agent.py."""

from __future__ import annotations

import json

from app.agents import planner_agent
from app.agents.planner_agent import planner_node
from app.auth.context import AuthContext
from app.constants.auth import ANONYMOUS_TENANT_ID, ANONYMOUS_USER_ID, ROLE_ANONYMOUS
from app.constants.workflow import (
    ERR_PLAN_LLM_FAILED,
    ERR_PLAN_SCHEMA_INVALID,
    ERR_PLAN_UNKNOWN_TOOL,
    NODE_PLANNER,
    WORKFLOW_STATUS_FAILED,
    WORKFLOW_STATUS_PENDING,
)
from app.llm.retry import LLMCallError


def _auth(anonymous: bool = False) -> AuthContext:
    if anonymous:
        return AuthContext(
            tenant_id=ANONYMOUS_TENANT_ID,
            user_id=ANONYMOUS_USER_ID,
            role=ROLE_ANONYMOUS,
            anonymous=True,
        )
    return AuthContext(tenant_id="t1", user_id="u1", role="user", anonymous=False)


def _state(message: str, *, anonymous: bool = False) -> dict:
    return {
        "messages": [{"role": "user", "content": message}],
        "auth": _auth(anonymous=anonymous),
        "request_id": "req-test",
        "session_id": "sess-test",
    }


def _plan_payload() -> dict:
    return {
        "task_type": "monitor_then_ticket",
        "steps": [
            {
                "id": "s1",
                "agent": "tool_agent",
                "purpose": "check weather",
                "tool": "get_weather",
                "args": {"city": "北京"},
            },
            {
                "id": "s2",
                "agent": "chat_agent",
                "purpose": "reply",
                "depends_on": ["s1"],
            },
        ],
        "compose_goal": "告诉用户结果",
    }


def test_planner_happy_path(llm_stub, monkeypatch) -> None:
    llm_stub.set_response(json.dumps(_plan_payload()))
    monkeypatch.setattr(planner_agent, "_new_plan_id", lambda: "plan-fixed")

    result = planner_node(_state("帮我查一下北京天气然后告诉用户"))

    assert result["workflow_status"] == WORKFLOW_STATUS_PENDING
    assert result["plan_id"] == "plan-fixed"
    plan = result["plan"]
    assert plan["task_type"] == "monitor_then_ticket"
    assert [s["id"] for s in plan["steps"]] == ["s1", "s2"]
    assert plan["steps"][0]["args"] == {"city": "北京"}
    assert plan["compose_goal"] == "告诉用户结果"
    assert result["agent_outputs"][NODE_PLANNER]["status"] == "ok"


def test_planner_llm_failure_is_fail_closed(llm_stub, monkeypatch) -> None:
    def _raise(*_a, **_kw):
        raise LLMCallError(
            code="timeout",
            message="boom",
            profile="routing",
            provider="deepseek",
            model="deepseek-chat",
        )

    # 直接替换 _call_planner_llm，验证节点在 LLM 失败时 fail-closed。
    monkeypatch.setattr(planner_agent, "_call_planner_llm", _raise)
    monkeypatch.setattr(planner_agent, "_new_plan_id", lambda: "plan-err")

    result = planner_node(_state("anything"))

    assert result["workflow_status"] == WORKFLOW_STATUS_FAILED
    assert result["plan"] == {}
    assert result["verification"]["unsupported_claims"] == [ERR_PLAN_LLM_FAILED]
    assert result["agent_outputs"][NODE_PLANNER]["error_code"] == ERR_PLAN_LLM_FAILED


def test_planner_schema_error_fails_closed(llm_stub) -> None:
    # 返回缺字段的半成品 JSON：parse_plan 必须 fail-closed。
    llm_stub.set_response('{"task_type": "x", "steps": []}')

    result = planner_node(_state("hello"))

    assert result["workflow_status"] == WORKFLOW_STATUS_FAILED
    assert result["verification"]["unsupported_claims"] == [ERR_PLAN_SCHEMA_INVALID]


def test_planner_unknown_tool_fails_closed(llm_stub) -> None:
    payload = _plan_payload()
    payload["steps"][0]["tool"] = "ghost.tool"
    llm_stub.set_response(json.dumps(payload))

    result = planner_node(_state("hello"))

    assert result["workflow_status"] == WORKFLOW_STATUS_FAILED
    assert result["verification"]["unsupported_claims"] == [ERR_PLAN_UNKNOWN_TOOL]
    assert "ghost.tool" in result["agent_outputs"][NODE_PLANNER]["detail"]


def test_planner_drops_requires_confirmation_from_output(llm_stub) -> None:
    payload = _plan_payload()
    payload["requires_confirmation"] = False  # Planner 不应声明
    payload["steps"][0]["requires_confirmation"] = False
    llm_stub.set_response(json.dumps(payload))

    result = planner_node(_state("hello"))

    assert result["workflow_status"] == WORKFLOW_STATUS_PENDING
    # plan dict 里不应该泄漏 requires_confirmation
    assert "requires_confirmation" not in result["plan"]
    for step in result["plan"]["steps"]:
        assert "requires_confirmation" not in step


def test_planner_prompt_hides_side_effect_tools_for_anonymous(llm_stub) -> None:
    llm_stub.set_response(json.dumps(_plan_payload()))

    planner_node(_state("hello", anonymous=True))

    # 必须至少一次 _create_chat_completion 调用：读取其 system prompt 并验证
    # ticket_create 相关业务名对匿名用户不可见。
    assert llm_stub.calls, "planner should call LLM exactly once"
    system_content = llm_stub.calls[0]["messages"][0]["content"]
    assert "ticket" not in system_content, system_content
    assert "get_weather" in system_content


def test_planner_generates_unique_plan_id(llm_stub) -> None:
    llm_stub.set_response(json.dumps(_plan_payload()))

    ids = {planner_node(_state("q"))["plan_id"] for _ in range(3)}
    assert len(ids) == 3

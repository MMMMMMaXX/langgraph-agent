"""Unit tests: app/workflow/schema.py + app/workflow/registry.py."""

from __future__ import annotations

import pytest

from app.constants.tooling import (
    TOOL_NAME_CALCULATE,
    TOOL_NAME_GET_WEATHER,
    TOOL_NAME_TICKET_CREATE,
)
from app.constants.workflow import (
    ERR_PLAN_ARGS_INVALID,
    ERR_PLAN_DAG_CYCLE,
    ERR_PLAN_SCHEMA_INVALID,
    ERR_PLAN_STEP_LIMIT,
    MAX_PLAN_STEPS,
)
from app.tools.metadata import ToolNotRegisteredError
from app.workflow import Plan, PlanValidationError, WorkflowStep, parse_plan
from app.workflow.registry import ToolRegistry, default_tool_registry

# ---- parse_plan / schema ---------------------------------------------------


def _basic_plan_dict() -> dict:
    return {
        "task_type": "weather_check",
        "steps": [
            {
                "id": "s1",
                "agent": "tool_agent",
                "purpose": "check weather",
                "tool": "get_weather",
                "args": {"city": "北京"},
            }
        ],
        "compose_goal": "告诉用户天气",
    }


def test_parse_plan_happy_path() -> None:
    plan = parse_plan(_basic_plan_dict())
    assert isinstance(plan, Plan)
    assert plan.task_type == "weather_check"
    assert plan.compose_goal == "告诉用户天气"
    assert plan.steps[0].id == "s1"
    assert plan.steps[0].args == {"city": "北京"}


def test_parse_plan_accepts_string_with_noise() -> None:
    raw = 'some prose before\n```json\n{"task_type":"t","steps":[{"id":"s1","agent":"chat_agent","purpose":"reply"}]}\n```'
    plan = parse_plan(raw)
    assert plan.steps[0].agent == "chat_agent"


def test_parse_plan_empty_payload_fails() -> None:
    with pytest.raises(PlanValidationError) as exc:
        parse_plan("")
    assert exc.value.code == ERR_PLAN_SCHEMA_INVALID


def test_parse_plan_silently_drops_requires_confirmation() -> None:
    payload = _basic_plan_dict()
    # Planner 不允许声明这个字段，schema 必须把它无声忽略。
    payload["requires_confirmation"] = True
    payload["steps"][0]["requires_confirmation"] = False
    plan = parse_plan(payload)
    assert not hasattr(plan, "requires_confirmation")
    assert not hasattr(plan.steps[0], "requires_confirmation")


def test_parse_plan_rejects_invalid_agent() -> None:
    payload = _basic_plan_dict()
    payload["steps"][0]["agent"] = "composer"
    with pytest.raises(PlanValidationError) as exc:
        parse_plan(payload)
    assert exc.value.code == ERR_PLAN_SCHEMA_INVALID


def test_parse_plan_rejects_tool_agent_without_tool() -> None:
    payload = _basic_plan_dict()
    del payload["steps"][0]["tool"]
    with pytest.raises(PlanValidationError):
        parse_plan(payload)


def test_parse_plan_rejects_rag_agent_without_query() -> None:
    payload = _basic_plan_dict()
    payload["steps"] = [{"id": "s1", "agent": "rag_agent", "purpose": "search"}]
    with pytest.raises(PlanValidationError):
        parse_plan(payload)


def test_parse_plan_rejects_chat_agent_with_tool() -> None:
    payload = _basic_plan_dict()
    payload["steps"] = [
        {"id": "s1", "agent": "chat_agent", "purpose": "reply", "tool": "get_weather"}
    ]
    with pytest.raises(PlanValidationError):
        parse_plan(payload)


def test_parse_plan_rejects_self_dependency() -> None:
    payload = _basic_plan_dict()
    payload["steps"][0]["depends_on"] = ["s1"]
    with pytest.raises(PlanValidationError) as exc:
        parse_plan(payload)
    assert exc.value.code == ERR_PLAN_DAG_CYCLE


def test_parse_plan_rejects_forward_reference() -> None:
    payload = {
        "task_type": "t",
        "steps": [
            {"id": "s1", "agent": "chat_agent", "purpose": "a", "depends_on": ["s2"]},
            {"id": "s2", "agent": "chat_agent", "purpose": "b"},
        ],
    }
    with pytest.raises(PlanValidationError) as exc:
        parse_plan(payload)
    assert exc.value.code == ERR_PLAN_DAG_CYCLE


def test_parse_plan_rejects_duplicate_ids() -> None:
    payload = {
        "task_type": "t",
        "steps": [
            {"id": "s1", "agent": "chat_agent", "purpose": "a"},
            {"id": "s1", "agent": "chat_agent", "purpose": "b"},
        ],
    }
    with pytest.raises(PlanValidationError):
        parse_plan(payload)


def test_parse_plan_rejects_bad_step_id() -> None:
    payload = {
        "task_type": "t",
        "steps": [{"id": "x1", "agent": "chat_agent", "purpose": "a"}],
    }
    with pytest.raises(PlanValidationError):
        parse_plan(payload)


def test_parse_plan_enforces_max_steps() -> None:
    payload = {
        "task_type": "t",
        "steps": [
            {"id": f"s{i}", "agent": "chat_agent", "purpose": "x"}
            for i in range(1, MAX_PLAN_STEPS + 2)
        ],
    }
    with pytest.raises(PlanValidationError) as exc:
        parse_plan(payload)
    assert exc.value.code == ERR_PLAN_STEP_LIMIT


def test_parse_plan_empty_steps_fails() -> None:
    with pytest.raises(PlanValidationError):
        parse_plan({"task_type": "t", "steps": []})


def test_parse_plan_args_error_code() -> None:
    # args 必须是 dict；给个字符串让 pydantic 抛 ValidationError，
    # parse_plan 应把错误码归为 ERR_PLAN_ARGS_INVALID。
    payload = _basic_plan_dict()
    payload["steps"][0]["args"] = "not a dict"
    with pytest.raises(PlanValidationError) as exc:
        parse_plan(payload)
    assert exc.value.code == ERR_PLAN_ARGS_INVALID


def test_workflow_step_frozen() -> None:
    step = WorkflowStep(id="s1", agent="chat_agent", purpose="x")
    with pytest.raises(Exception):
        step.id = "s2"  # type: ignore[misc]


# ---- ToolRegistry ----------------------------------------------------------


def test_registry_resolves_dotted_business_name() -> None:
    # 点号按"直接替换为下划线后在 registry 里查"的规则解析。
    assert default_tool_registry.resolve("ticket.create") == TOOL_NAME_TICKET_CREATE


def test_registry_resolves_canonical_function_name() -> None:
    assert default_tool_registry.resolve(TOOL_NAME_CALCULATE) == TOOL_NAME_CALCULATE
    assert default_tool_registry.resolve(TOOL_NAME_GET_WEATHER) == TOOL_NAME_GET_WEATHER


def test_registry_unknown_tool_raises() -> None:
    with pytest.raises(ToolNotRegisteredError):
        default_tool_registry.resolve("no.such.tool")


def test_registry_has_helper() -> None:
    assert default_tool_registry.has("ticket.create")
    assert not default_tool_registry.has("")


def test_registry_visible_tools_strips_side_effect_for_anonymous() -> None:
    from app.auth.context import AuthContext
    from app.constants.auth import (
        ANONYMOUS_TENANT_ID,
        ANONYMOUS_USER_ID,
        ROLE_ANONYMOUS,
    )

    auth = AuthContext(
        tenant_id=ANONYMOUS_TENANT_ID,
        user_id=ANONYMOUS_USER_ID,
        role=ROLE_ANONYMOUS,
        anonymous=True,
    )
    visible_names = {m.name for m in default_tool_registry.visible_tools(auth)}
    assert TOOL_NAME_TICKET_CREATE not in visible_names
    assert TOOL_NAME_GET_WEATHER in visible_names


def test_registry_visible_tools_keeps_side_effect_for_user() -> None:
    from app.auth.context import AuthContext

    auth = AuthContext(tenant_id="t1", user_id="u1", role="user", anonymous=False)
    visible_names = {m.name for m in default_tool_registry.visible_tools(auth)}
    assert TOOL_NAME_TICKET_CREATE in visible_names


def test_registry_injectable_for_tests() -> None:
    """构造裁剪 registry 只暴露 get_weather，用于 planner 测试场景。"""
    from app.tools.metadata import TOOL_METADATA

    scoped = ToolRegistry(
        metadata={TOOL_NAME_GET_WEATHER: TOOL_METADATA[TOOL_NAME_GET_WEATHER]}
    )
    assert scoped.has(TOOL_NAME_GET_WEATHER)
    assert not scoped.has(TOOL_NAME_TICKET_CREATE)

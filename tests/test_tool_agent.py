"""`tool_agent_node` 测试。

分两层：
1. **pure-compute 单测**：`should_finalize_with_llm` / `format_single_tool_answer`。
   无外部依赖，纯字符串逻辑。
2. **端到端 node 测试**：用 `llm_stub` 按 `trace_stage` 分派，模拟 LLM 的
   function-calling 响应，驱动 `tool_agent_node` 走完 tool_select → 执行工具 →
   （可选）tool_finalize 的整条链路。

关键点：
- tool_select 阶段 stub 返回 `{"tool_calls": [...]}` dict；
- tool_finalize 阶段 stub 返回普通 str；
- 工具实现用的是 `app.tools.tools.get_weather`（查表）/ `calculate`（AST 求值），
  天然无 I/O，不需要额外 mock。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from app.agents.tool_agent import (
    format_single_tool_answer,
    should_finalize_with_llm,
    tool_agent_node,
)
from app.auth.context import AuthContext
from app.constants.auth import (
    ANONYMOUS_TENANT_ID,
    ANONYMOUS_USER_ID,
    ROLE_ANONYMOUS,
    ROLE_USER,
)
from app.constants.routes import ROUTE_TOOL_AGENT
from app.constants.tool_safety import (
    CONFIRMATION_SECRET_ENV,
    TOOL_STATUS_SUCCEEDED,
)
from app.constants.tooling import (
    TOOL_NAME_CALCULATE,
    TOOL_NAME_GET_WEATHER,
    TOOL_NAME_TICKET_CREATE,
    TOOL_TYPE_NONE,
)
from app.tools import execution_record as exec_record_mod
from app.tools import ticket as ticket_mod
from app.tools.confirmation import issue_token
from app.tools.idempotency import compute_idempotency_key
from tests.conftest import make_tool_call

# ----------------------------- pure-compute 单测 -----------------------------


def test_should_finalize_detects_multi_intent_keywords() -> None:
    # 任意一个多意图关键词出现都应触发 finalize
    assert should_finalize_with_llm("北京天气，顺便算一下 1+1")
    assert should_finalize_with_llm("另外再帮我查下上海")
    assert should_finalize_with_llm("同时计算 2*3")


def test_should_finalize_single_intent_skips_finalize() -> None:
    # 单一意图不触发，避免浪费二次 LLM 成本
    assert not should_finalize_with_llm("北京天气")
    assert not should_finalize_with_llm("1+2 等于多少")
    assert not should_finalize_with_llm("")


def test_format_single_tool_answer_per_tool_type() -> None:
    # weather 原样返回；calculate 包装成"计算结果是 X。"；未知工具原样兜底
    assert format_single_tool_answer(TOOL_NAME_GET_WEATHER, "北京晴") == "北京晴"
    assert format_single_tool_answer(TOOL_NAME_CALCULATE, "3") == "计算结果是 3。"
    assert format_single_tool_answer("unknown_tool", "raw") == "raw"


# -------------------------- tool_agent_node 端到端 --------------------------


def _state(message: str) -> dict:
    return {
        "messages": [{"role": "user", "content": message}],
        "auth": AuthContext(
            tenant_id=ANONYMOUS_TENANT_ID,
            user_id=ANONYMOUS_USER_ID,
            groups=(),
            role=ROLE_ANONYMOUS,
            anonymous=True,
        ),
    }


def test_tool_agent_single_weather_happy_path(llm_stub) -> None:
    # LLM 选 get_weather(city="北京")；单工具、单意图 → 跳过 tool_finalize
    def stub_fn(trace_stage: str = "", **_):
        if trace_stage == "tool_select":
            return {
                "tool_calls": [make_tool_call("get_weather", {"city": "北京"})],
            }
        # 不该走到 tool_finalize
        raise AssertionError(f"unexpected trace_stage: {trace_stage}")

    llm_stub.set_response_fn(stub_fn)

    result = tool_agent_node(_state("北京天气怎么样"))

    # answer 来自工具模板化输出（weather 原样）
    assert result["answer"]
    assert result["agent_outputs"][ROUTE_TOOL_AGENT] == result["answer"]

    debug = result["debug_info"][ROUTE_TOOL_AGENT]
    assert debug["tool_type"] == TOOL_NAME_GET_WEATHER
    assert debug["finalize_with_llm"] is False
    assert len(debug["tool_calls"]) == 1
    assert debug["tool_calls"][0]["name"] == TOOL_NAME_GET_WEATHER
    assert debug["tool_calls"][0]["arguments"] == {"city": "北京"}
    assert debug["error"] == ""


def test_tool_agent_single_calculate_happy_path(llm_stub) -> None:
    # 单工具 calculate，走模板化返回
    llm_stub.set_response_fn(
        lambda trace_stage="", **_: (
            {"tool_calls": [make_tool_call("calculate", {"expression": "1+2"})]}
            if trace_stage == "tool_select"
            else ""
        )
    )

    result = tool_agent_node(_state("1+2 等于多少"))

    # calculate 的模板：`计算结果是 X。`，AST 求值 1+2=3
    assert result["answer"] == "计算结果是 3。"
    debug = result["debug_info"][ROUTE_TOOL_AGENT]
    assert debug["tool_type"] == TOOL_NAME_CALCULATE
    assert debug["finalize_with_llm"] is False


def test_tool_agent_multi_intent_triggers_llm_finalize(llm_stub) -> None:
    # 多意图关键词（"顺便"）触发 finalize，LLM 二次调用整合
    finalize_called = {"hit": False}

    def stub_fn(trace_stage: str = "", **_):
        if trace_stage == "tool_select":
            return {
                "tool_calls": [
                    make_tool_call("get_weather", {"city": "北京"}, call_id="c1"),
                    make_tool_call("calculate", {"expression": "1+1"}, call_id="c2"),
                ],
            }
        if trace_stage == "tool_finalize":
            finalize_called["hit"] = True
            return "综合回答：北京今天阴，1+1=2"
        return ""

    llm_stub.set_response_fn(stub_fn)

    result = tool_agent_node(_state("北京天气怎么样，顺便算一下 1+1"))

    assert finalize_called["hit"], "multi-intent 必须触发 tool_finalize"
    assert result["answer"] == "综合回答：北京今天阴，1+1=2"
    debug = result["debug_info"][ROUTE_TOOL_AGENT]
    assert debug["finalize_with_llm"] is True
    assert len(debug["tool_calls"]) == 2
    # tool_type 会拼接所有工具名
    assert "get_weather" in debug["tool_type"]
    assert "calculate" in debug["tool_type"]


def test_tool_agent_no_tool_selected_uses_llm_content(llm_stub) -> None:
    # LLM 不选工具，直接文本回答 → answer 来自第一次响应的 content
    llm_stub.set_response_fn(
        lambda trace_stage="", **_: (
            {"content": "这问题我直接答了：42"} if trace_stage == "tool_select" else ""
        )
    )

    result = tool_agent_node(_state("人生的意义是什么"))

    assert result["answer"] == "这问题我直接答了：42"
    debug = result["debug_info"][ROUTE_TOOL_AGENT]
    assert debug["tool_type"] == TOOL_TYPE_NONE
    assert debug["tool_calls"] == []


def test_tool_agent_unknown_tool_produces_error_output(llm_stub) -> None:
    # LLM 选了一个不在 TOOL_IMPLS 里的工具 → chat_with_tools 兜底输出
    # "工具 X 不存在。"；单 tool_result 走 format_single_tool_answer 的
    # fallback 分支原样返回
    llm_stub.set_response_fn(
        lambda trace_stage="", **_: (
            {"tool_calls": [make_tool_call("nonexistent_tool", {})]}
            if trace_stage == "tool_select"
            else ""
        )
    )

    result = tool_agent_node(_state("随便什么"))

    assert "nonexistent_tool" in result["answer"]
    assert "不存在" in result["answer"]
    debug = result["debug_info"][ROUTE_TOOL_AGENT]
    assert debug["tool_results"][0]["name"] == "nonexistent_tool"


def test_tool_agent_llm_exception_returns_friendly_message(llm_stub) -> None:
    # LLM 调用抛异常（网络、超时等）→ node 兜底返回友好提示，不冒泡
    def exploding(**_):
        raise RuntimeError("network down")

    llm_stub.set_response_fn(exploding)

    result = tool_agent_node(_state("北京天气"))

    assert result["answer"] == "工具调用暂时失败，请稍后再试。"
    debug = result["debug_info"][ROUTE_TOOL_AGENT]
    assert debug["error"]  # build_error_info 会填充错误信息
    assert debug["tool_type"] == TOOL_TYPE_NONE


# -------------------------- side_effect pipeline 集成 --------------------------


@dataclass(frozen=True)
class _TmpOpsConfig:
    path: str


@pytest.fixture()
def side_effect_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """给 side_effect 工具链隔离 DB + 注入 confirmation secret。"""

    cfg = _TmpOpsConfig(path=str(tmp_path / "operations.sqlite3"))
    monkeypatch.setattr(exec_record_mod, "OPERATIONS_CONFIG", cfg)
    monkeypatch.setattr(ticket_mod, "OPERATIONS_CONFIG", cfg)
    monkeypatch.setenv(CONFIRMATION_SECRET_ENV, "tool-agent-test-secret")


def _authed_state(message: str, *, confirmation_token: str = "") -> dict:
    return {
        "session_id": "s-agent",
        "request_id": "req-1",
        "messages": [{"role": "user", "content": message}],
        "auth": AuthContext(
            tenant_id="t1",
            user_id="u1",
            groups=(),
            role=ROLE_USER,
            anonymous=False,
        ),
        "confirmation_token": confirmation_token,
    }


def test_tool_agent_ticket_create_without_token_returns_pending(
    llm_stub, side_effect_env
) -> None:
    """首次请求：LLM 选 ticket_create → pipeline 签发 token，node 回填 state。"""

    llm_stub.set_response_fn(
        lambda trace_stage="", **_: (
            {
                "tool_calls": [
                    make_tool_call(TOOL_NAME_TICKET_CREATE, {"title": "Reset password"})
                ]
            }
            if trace_stage == "tool_select"
            else ""
        )
    )

    result = tool_agent_node(_authed_state("帮我创建一个工单"))

    # answer 来自 pipeline 的 need_confirmation 提示文本。
    assert "确认" in result["answer"]
    # state 里带着可供客户端回填的 token + 意图摘要。
    pc = result["pending_confirmation"]
    assert pc["tool_name"] == TOOL_NAME_TICKET_CREATE
    assert pc["args"] == {"title": "Reset password"}
    assert pc["token"]
    # 首次请求还没执行工具，executions 为空。
    assert result["tool_executions"] == []


def test_tool_agent_ticket_create_with_valid_token_executes(
    llm_stub, side_effect_env
) -> None:
    """二次请求携带有效 token → pipeline 执行 ticket_create 并写入 execution。"""

    args = {"title": "Fix login"}
    state = _authed_state("执行工单创建")
    ikey = compute_idempotency_key(
        tenant_id=state["auth"].tenant_id,
        user_id=state["auth"].user_id,
        session_id=state["session_id"],
        tool_name=TOOL_NAME_TICKET_CREATE,
        args=args,
    )
    token, _ = issue_token(
        idempotency_key=ikey,
        tool_name=TOOL_NAME_TICKET_CREATE,
        tenant_id=state["auth"].tenant_id,
        user_id=state["auth"].user_id,
        args=args,
    )
    state["confirmation_token"] = token

    llm_stub.set_response_fn(
        lambda trace_stage="", **_: (
            {"tool_calls": [make_tool_call(TOOL_NAME_TICKET_CREATE, args)]}
            if trace_stage == "tool_select"
            else ""
        )
    )

    result = tool_agent_node(state)

    # answer 是 ticket_create 返回的 dict 的 repr，至少包含 ticket_id。
    assert "TK-" in result["answer"]
    # 不再需要确认。
    assert result["pending_confirmation"] == {}
    # 写了一条 succeeded 的 execution record。
    assert len(result["tool_executions"]) == 1
    assert result["tool_executions"][0]["status"] == TOOL_STATUS_SUCCEEDED
    assert result["tool_executions"][0]["tool_name"] == TOOL_NAME_TICKET_CREATE


def test_tool_agent_ticket_create_filtered_out_for_anonymous(
    llm_stub, side_effect_env
) -> None:
    """匿名身份：ticket_create 从 tools 列表里被过滤掉，LLM 看不到更不会调。

    这里通过捕获 stub 收到的 tools 参数反推过滤生效（直接断言列表里没有
    ticket_create）。
    """

    captured_tools: list = []

    def stub_fn(trace_stage: str = "", **kwargs):
        if trace_stage == "tool_select":
            # 记录 LLM 看到的工具列表。
            captured_tools.extend(kwargs.get("messages", []))  # 兼容不同调用形态
        return {"content": "匿名回复"}

    # 上面的 kwargs 并不直接含 tools；改走 chat_with_tools 的 monkeypatch 更可靠。
    import app.agents.tool_agent as ta_mod

    original = ta_mod.chat_with_tools

    def fake_chat_with_tools(*, tools, tool_impls, **_kwargs):
        captured_tools.append([t["function"]["name"] for t in tools])
        return {"tool_calls": [], "tool_results": [], "answer": "匿名回复"}

    ta_mod.chat_with_tools = fake_chat_with_tools  # type: ignore[assignment]
    try:
        state = {
            "session_id": "s-anon",
            "request_id": "req-1",
            "messages": [{"role": "user", "content": "帮我开一个工单"}],
            "auth": AuthContext(
                tenant_id=ANONYMOUS_TENANT_ID,
                user_id=ANONYMOUS_USER_ID,
                groups=(),
                role=ROLE_ANONYMOUS,
                anonymous=True,
            ),
            "confirmation_token": "",
        }
        tool_agent_node(state)
    finally:
        ta_mod.chat_with_tools = original

    assert captured_tools, "chat_with_tools 必须被调用一次"
    tool_names = captured_tools[0]
    assert TOOL_NAME_TICKET_CREATE not in tool_names
    # read_only 工具应当保留。
    assert TOOL_NAME_GET_WEATHER in tool_names
    assert TOOL_NAME_CALCULATE in tool_names

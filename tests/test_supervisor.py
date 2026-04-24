"""Supervisor 路由决策测试。

测试目标：`supervisor_node(state)` 的路由规则与 LLM fallback 行为。

分两批：
1. **规则层**：不触发 LLM，纯走关键词匹配。断言 routes / intent / debug_info。
2. **LLM fallback**：规则都不命中时，supervisor 会调 `plan_routes(message)`。
   这里用 monkeypatch 替换 `plan_routes`，验证：
   (a) fallback 真的被走到了
   (b) 返回值被正确传递到 state

注意：supervisor 通过 `from app.llm import plan_routes` 导入，在模块加载时就
捕获了函数引用。所以要 patch `app.agents.supervisor.plan_routes`，而不是
`app.llm.chat.plan_routes`——后者改了对调用处无效。
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from app.agents.supervisor import supervisor_node
from app.constants.routes import (
    ROUTE_CHAT_AGENT,
    ROUTE_NOVEL_SCRIPT_AGENT,
    ROUTE_RAG_AGENT,
    ROUTE_TOOL_AGENT,
)


def _state_with_message(message: str, history: list[dict] | None = None) -> dict:
    """造一个最小 AgentState：只带 messages，其余字段走默认。"""
    messages = list(history or [])
    messages.append({"role": "user", "content": message})
    return {"messages": messages}


# ----------------------------- 规则层（不需要 LLM） -----------------------------


def test_script_task_routes_to_novel_agent() -> None:
    # "改成剧本" 命中 looks_like_script_task，优先级最高。
    # 这条规则存在就是为了防止小说正文里碰巧出现"之前"字样被误路由到 chat。
    result = supervisor_node(_state_with_message("把下面这段改成剧本：某年某月……"))
    assert result["routes"] == [ROUTE_NOVEL_SCRIPT_AGENT]
    assert result["intent"] == "creative"
    assert result["debug_info"]["supervisor"]["route_reason"]


def test_meta_history_query_routes_to_chat_agent() -> None:
    # "刚刚" 命中 META_HISTORY_QUERY_KEYWORDS，且消息短于 80 字 → chat
    result = supervisor_node(_state_with_message("刚刚我问了什么"))
    assert result["routes"] == [ROUTE_CHAT_AGENT]
    assert result["intent"] == "chat"


def test_weather_query_routes_to_tool_agent() -> None:
    # "天气" 命中 WEATHER_QUERY_KEYWORDS，不含知识关键词 → 只走 tool
    result = supervisor_node(_state_with_message("北京天气怎么样"))
    assert result["routes"] == [ROUTE_TOOL_AGENT]
    assert result["intent"] == "tool"


def test_knowledge_query_routes_to_rag_agent() -> None:
    # "是什么" 命中 KNOWLEDGE_QUERY_KEYWORDS，不含天气/数学 → 只走 rag
    result = supervisor_node(_state_with_message("WAI-ARIA 是什么"))
    assert result["routes"] == [ROUTE_RAG_AGENT]
    assert result["intent"] == "retrieval"


def test_weather_plus_knowledge_produces_hybrid_route() -> None:
    # "北京天气是什么意思" 同时触发 weather + knowledge，
    # 验证两个路由并存 + intent = hybrid
    result = supervisor_node(_state_with_message("北京天气是什么意思"))
    assert set(result["routes"]) == {ROUTE_TOOL_AGENT, ROUTE_RAG_AGENT}
    assert result["intent"] == "hybrid"


# ----------------------------- LLM fallback -----------------------------


@pytest.fixture
def plan_routes_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[list[str]], None]:
    """把 supervisor 里的 plan_routes 换成可控桩。

    返回一个 setter：调一次决定下一次 plan_routes() 的返回值。
    """
    planned: list[str] = [ROUTE_CHAT_AGENT]  # 默认兜底

    def _stub(_message: str) -> list[str]:
        return list(planned)

    # 必须打到 supervisor 模块上，因为它在 import 时就把 plan_routes 引用绑死了
    import app.agents.supervisor as sup

    monkeypatch.setattr(sup, "plan_routes", _stub)

    def _set(routes: list[str]) -> None:
        planned.clear()
        planned.extend(routes)

    return _set


def test_llm_fallback_when_no_rule_matches(
    plan_routes_stub: Callable[[list[str]], None],
) -> None:
    # "hello" 不命中任何规则 → 应走 LLM fallback，拿到 stub 里的结果
    plan_routes_stub([ROUTE_RAG_AGENT])

    result = supervisor_node(_state_with_message("hello"))

    assert result["routes"] == [ROUTE_RAG_AGENT]
    assert result["intent"] == "retrieval"
    assert result["debug_info"]["supervisor"]["route_reason"] == "llm fallback"


def test_llm_fallback_not_called_when_rule_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 规则命中时 plan_routes 不应该被调用（省成本 + 稳定性）。
    # 用一个会抛异常的桩：一旦被调就炸，直接暴露逻辑错误。
    import app.agents.supervisor as sup

    def _explode(_message: str) -> list[str]:
        raise AssertionError("plan_routes should not be invoked when rule matches")

    monkeypatch.setattr(sup, "plan_routes", _explode)

    # "天气" 命中规则，不应触发 fallback
    result = supervisor_node(_state_with_message("北京天气怎么样"))
    assert result["routes"] == [ROUTE_TOOL_AGENT]


# --------------------- 端到端 LLM fallback（走真实调用链） ---------------------
#
# 上面两条 fallback 测试直接 patch 了 `supervisor.plan_routes`，只覆盖到
# supervisor 这一层。这里换一种姿势：用顶层 `llm_stub` 拦截底层的
# `_create_chat_completion`，让 supervisor → `plan_routes` → SDK 整条链路真实跑起来，
# 顺带验证 llm_stub 在"多层薄包装"场景下也能稳定接管。


def test_llm_fallback_via_llm_stub_end_to_end(llm_stub) -> None:
    # plan_routes 期望 LLM 返回一个 JSON array 字符串，再解析出合法 agent 名字
    llm_stub.set_response('["rag_agent"]')

    # "hello" 不命中任何规则 → 真实 plan_routes() → 被桩拦截到的 _create_chat_completion
    result = supervisor_node(_state_with_message("hello"))

    assert result["routes"] == [ROUTE_RAG_AGENT]
    assert result["intent"] == "retrieval"
    assert result["debug_info"]["supervisor"]["route_reason"] == "llm fallback"

    # 断言真的走到了 LLM 层：trace_stage 由 plan_routes 显式传入，容易对上号
    assert any(
        call.get("trace_stage") == "plan_routes" for call in llm_stub.calls
    ), f"expected a plan_routes LLM call, got: {llm_stub.calls!r}"


def test_llm_fallback_invalid_json_defaults_to_chat(llm_stub) -> None:
    # LLM 返回无法解析的内容时，plan_routes 应降级到 [chat_agent]，
    # 避免把异常冒泡到 supervisor 搞出 5xx。
    llm_stub.set_response("not a json at all")

    result = supervisor_node(_state_with_message("some truly unknown request"))

    assert result["routes"] == [ROUTE_CHAT_AGENT]
    assert result["intent"] == "chat"

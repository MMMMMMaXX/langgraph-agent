"""LangGraph 端到端测试：驱动完整的 supervisor → agent → memory 流水。

测试关注"组装"层面的正确性，具体每个 agent 的内部行为已由各自的单测覆盖：
- 图的节点/边接上了吗？supervisor 的路由决定真的把状态交到对的 agent 手里？
- state 合并（debug_info / node_timings / answer / messages）没掉字段？
- memory_node 能把 answer 接住 + 追加 assistant 消息？

I/O 隔离策略：
1. `LANGGRAPH_CHECKPOINT_ENABLED=false` → 用 InMemorySaver，不写 sqlite 文件
2. llm_stub 接管所有 LLM 调用
3. memory_node 里的 Chroma / sqlite 写入：monkeypatch `add_memory_item` +
   `append_history_event` 到空操作
4. RAG pipeline：monkeypatch 三个子阶段到 dataclass 桩

不测 chat_agent 完整路径的理由：chat_agent 底层依赖 Chroma 向量查询 + 事实提取，
I/O 隔离层次多，收益比不如专注于"图拼接"这个真正的 E2E 价值点。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from app.agents.rag.types import (
    DocRetrievalResult,
    MemoryRetrievalResult,
    RewriteResult,
)
from app.chat_service import create_initial_state, run_chat_turn
from app.constants.routes import (
    NODE_MEMORY,
    NODE_SUPERVISOR,
    ROUTE_RAG_AGENT,
    ROUTE_TOOL_AGENT,
)
from tests.conftest import make_tool_call


@pytest.fixture
def isolate_memory_io(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """阻断 memory_node 的 Chroma + sqlite/jsonl 写入，返回调用计数器。

    注意：`from ... import add_memory_item` / `append_history_event` 把函数绑到了
    `app.nodes.memory` 的模块命名空间，所以要打到这个模块上，不是源模块。
    """
    import app.nodes.memory as mem_mod

    counters: dict[str, Any] = {"add_memory": 0, "append_history": 0}

    def fake_add_memory(*_args, **_kwargs) -> None:
        counters["add_memory"] += 1

    def fake_append_history(**kwargs) -> dict:
        counters["append_history"] += 1
        # 返回 service 层契约：dict with rewritten_query / user_message / skipped_duplicate
        return {
            "skipped_duplicate": False,
            "rewritten_query": kwargs.get("rewritten_query", ""),
            "user_message": kwargs.get("user_message", ""),
        }

    monkeypatch.setattr(mem_mod, "add_memory_item", fake_add_memory)
    monkeypatch.setattr(mem_mod, "append_history_event", fake_append_history)
    yield counters


@pytest.fixture
def isolate_rag_pipelines(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict]:
    """把 rag node 的三个子阶段打桩成可控返回。

    默认：文档命中 1 条。测试如需不同场景，改 `captured['doc_result']` 再调用。
    """
    import app.agents.rag.node as rag_mod

    captured: dict = {
        "doc_result": DocRetrievalResult(
            docs=[{"id": "d1", "content": "doc", "score": 0.9}],
            filtered_docs=[{"id": "d1", "content": "doc", "score": 0.9}],
            doc_hits=[{"id": "d1", "content": "doc", "score": 0.9}],
            merged_doc_hits=[
                {"id": "d1", "content": "doc hit content", "score": 0.9}
            ],
            retrieval_debug={},
            errors=[],
            timings_ms={},
        ),
        "memory_result": MemoryRetrievalResult(
            memory_hits=[],
            memory_before_rerank=[],
            retrieval_debug={},
            errors=[],
            timings_ms={},
        ),
    }

    monkeypatch.setattr(
        rag_mod,
        "rewrite_rag_query",
        lambda message, messages, summary: RewriteResult(
            query=message, errors=[], timing_ms=0.1
        ),
    )
    monkeypatch.setattr(
        rag_mod, "retrieve_docs_for_rag", lambda _q: captured["doc_result"]
    )
    monkeypatch.setattr(
        rag_mod,
        "retrieve_memory_for_rag",
        lambda _q, *, session_id, enabled: captured["memory_result"],
    )
    yield captured


# ------------------------------- tool_agent E2E -------------------------------


def test_e2e_weather_query_routes_through_tool_agent(
    llm_stub, isolate_memory_io
) -> None:
    """用户问天气 → supervisor 规则命中 → tool_agent → memory_node → 答案回填。"""

    def stub(trace_stage: str = "", **_):
        if trace_stage == "tool_select":
            return {"tool_calls": [make_tool_call("get_weather", {"city": "北京"})]}
        return ""  # 其他阶段（tool_finalize / summarize 等）没走到就无所谓

    llm_stub.set_response_fn(stub)

    state = create_initial_state("sess-tool")
    state["request_id"] = "req-1"
    result = run_chat_turn(state, "北京天气怎么样")

    # 1. answer 非空 + routes 指向 tool_agent
    assert result["answer"]
    assert result["routes"] == [ROUTE_TOOL_AGENT]

    # 2. debug_info 来自多个节点的合并
    assert NODE_SUPERVISOR in result["debug_info"]
    assert ROUTE_TOOL_AGENT in result["debug_info"]
    assert NODE_MEMORY in result["debug_info"]
    # supervisor 记录路由理由
    assert result["debug_info"][NODE_SUPERVISOR]["route_reason"]
    # tool_agent 记录实际调用
    tool_debug = result["debug_info"][ROUTE_TOOL_AGENT]
    assert tool_debug["tool_calls"][0]["name"] == "get_weather"

    # 3. node_timings 至少覆盖走过的节点
    timings = result["node_timings"]
    assert timings[NODE_SUPERVISOR] >= 0
    assert timings[ROUTE_TOOL_AGENT] >= 0
    assert timings[NODE_MEMORY] >= 0

    # 4. messages 追加了 assistant 回复
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"] == result["answer"]

    # 5. memory_node 触发了 history 写入（单轮短消息会走兜底）
    # 具体写不写由 decide_memory_write 决定，这里只验证 hook 被调用过
    # 至少 append_history 应该被尝试过（单轮非 meta 查询）
    assert isolate_memory_io["append_history"] >= 1


# ------------------------------- rag_agent E2E -------------------------------


def test_e2e_knowledge_query_routes_through_rag_agent(
    llm_stub, isolate_memory_io, isolate_rag_pipelines
) -> None:
    """知识类问题（"是什么"）→ supervisor 规则命中 rag_agent → 检索到文档 → 答案生成。"""

    llm_stub.set_response("WAI-ARIA 是一套无障碍规范，用来给网页加语义标签。")

    state = create_initial_state("sess-rag")
    state["request_id"] = "req-2"
    result = run_chat_turn(state, "WAI-ARIA 是什么")

    assert result["routes"] == [ROUTE_RAG_AGENT]
    assert result["answer"].startswith("WAI-ARIA")

    # rag_agent 的 debug_info 应记录 doc_used + rewritten_query
    rag_debug = result["debug_info"][ROUTE_RAG_AGENT]
    assert rag_debug["doc_used"] is True
    assert rag_debug["rewritten_query"]

    # messages 闭环
    assert result["messages"][-1]["role"] == "assistant"


def test_e2e_rag_no_hits_falls_back_to_insufficient(
    llm_stub, isolate_memory_io, isolate_rag_pipelines
) -> None:
    """文档 + 记忆都无命中 → answer = "资料不足"；链路跑完不崩。"""

    # 清空 doc_result → 触发空文档分支
    isolate_rag_pipelines["doc_result"] = DocRetrievalResult(
        docs=[], filtered_docs=[], doc_hits=[], merged_doc_hits=[],
        retrieval_debug={}, errors=[], timings_ms={},
    )

    state = create_initial_state("sess-rag-empty")
    result = run_chat_turn(state, "一个没人知道答案的问题是什么")

    assert result["routes"] == [ROUTE_RAG_AGENT]
    assert result["answer"] == "资料不足"
    # answer 仍然被 memory_node 处理：assistant 消息被追加
    assert result["messages"][-1]["content"] == "资料不足"


# ------------------------------ 多轮状态 E2E ------------------------------


def test_e2e_multi_turn_preserves_messages(
    llm_stub, isolate_memory_io
) -> None:
    """连续两轮：第二轮的入参 state.messages 应该带上第一轮的 user + assistant。"""

    def stub(trace_stage: str = "", **_):
        if trace_stage == "tool_select":
            return {"tool_calls": [make_tool_call("get_weather", {"city": "北京"})]}
        return ""

    llm_stub.set_response_fn(stub)

    # 第一轮
    state = create_initial_state("sess-multi")
    state["request_id"] = "req-t1"
    r1 = run_chat_turn(state, "北京天气")
    assert r1["messages"][-1]["role"] == "assistant"

    # 第二轮：把上一轮的 state 喂回去（InMemorySaver 下也能从 checkpoint 读）
    r2_input = create_initial_state("sess-multi")
    r2_input["request_id"] = "req-t2"
    r2_input["messages"] = r1["messages"]  # 显式传递（checkpoint 也会兜）
    r2_input["summary"] = r1["summary"]

    r2 = run_chat_turn(r2_input, "上海呢")

    # 第二轮的 messages 应该是 user1 + asst1 + user2 + asst2
    assert len(r2["messages"]) == 4
    roles = [m["role"] for m in r2["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert r2["messages"][2]["content"] == "上海呢"

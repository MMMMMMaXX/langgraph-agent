from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from app.agents.chat_agent import chat_agent_node
from app.agents.composer_agent import composer_node
from app.agents.merge import merge_node
from app.agents.novel_script_agent import novel_script_agent_node
from app.agents.planner_agent import planner_node
from app.agents.rag_agent import rag_agent_node
from app.agents.supervisor import supervisor_node
from app.agents.tool_agent import tool_agent_node
from app.agents.verifier_agent import verifier_node
from app.agents.workflow_executor import workflow_executor_node
from app.checkpointing.factory import get_checkpointer
from app.constants.routes import (
    NODE_MEMORY,
    NODE_MERGE,
    NODE_SUPERVISOR,
    ROUTE_CHAT_AGENT,
    ROUTE_NOVEL_SCRIPT_AGENT,
    ROUTE_RAG_AGENT,
    ROUTE_TOOL_AGENT,
    ROUTE_WORKFLOW,
)
from app.constants.workflow import (
    NODE_COMPOSER,
    NODE_PLANNER,
    NODE_VERIFIER,
    NODE_WORKFLOW_EXECUTOR,
)
from app.nodes.memory import memory_node
from app.runtime_context import get_stream_callback
from app.state import AgentState
from app.utils.logger import log_node_timing, now_ms


def route_after_supervisor(state: AgentState) -> str:
    routes = state.get("routes", [])

    if routes == [ROUTE_CHAT_AGENT]:
        return ROUTE_CHAT_AGENT

    if routes == [ROUTE_NOVEL_SCRIPT_AGENT]:
        return ROUTE_NOVEL_SCRIPT_AGENT

    # Phase 2：workflow 分支走 Planner → Executor → Merge。
    # 走独立路由是因为 Planner 会失败 fail-closed，不能和 tool/rag 同级混用。
    if routes == [ROUTE_WORKFLOW]:
        return NODE_PLANNER

    if ROUTE_TOOL_AGENT in routes:
        return ROUTE_TOOL_AGENT

    if ROUTE_RAG_AGENT in routes:
        return ROUTE_RAG_AGENT

    return ROUTE_CHAT_AGENT


def route_after_tool(state: AgentState) -> str:
    routes = state.get("routes", [])
    if ROUTE_RAG_AGENT in routes:
        return ROUTE_RAG_AGENT
    return NODE_MERGE


def with_timing(
    name: str, node_fn: Callable[[AgentState], AgentState]
) -> Callable[[AgentState], AgentState]:
    def timed_node(state: AgentState) -> AgentState:
        started_at_ms = now_ms()
        stream_callback = get_stream_callback()
        if callable(stream_callback):
            stream_callback(
                "node_started",
                {
                    "node": name,
                    "request_id": state.get("request_id", ""),
                    "session_id": state.get("session_id", "default"),
                },
            )

        result = node_fn(state)
        duration_ms = now_ms() - started_at_ms

        updated_result: AgentState = {
            **result,
            "request_id": result.get("request_id", state.get("request_id", "")),
            "session_id": result.get("session_id", state.get("session_id", "default")),
            "streamed_answer": result.get(
                "streamed_answer", state.get("streamed_answer", False)
            ),
        }
        updated_result["node_timings"] = {name: round(duration_ms, 2)}

        stream_callback = get_stream_callback()
        if callable(stream_callback):
            stream_callback(
                "node_completed",
                {
                    "node": name,
                    "request_id": updated_result.get("request_id", ""),
                    "session_id": updated_result.get("session_id", "default"),
                    "duration_ms": round(duration_ms, 2),
                    "debug": updated_result.get("debug_info", {}).get(name, {}),
                },
            )

        log_node_timing(
            name=name,
            duration_ms=duration_ms,
            request_id=updated_result.get("request_id", ""),
            session_id=updated_result.get("session_id", ""),
        )
        return updated_result

    return timed_node


builder = StateGraph(AgentState)

builder.add_node(NODE_SUPERVISOR, with_timing(NODE_SUPERVISOR, supervisor_node))
builder.add_node(ROUTE_TOOL_AGENT, with_timing(ROUTE_TOOL_AGENT, tool_agent_node))
builder.add_node(ROUTE_RAG_AGENT, with_timing(ROUTE_RAG_AGENT, rag_agent_node))
builder.add_node(ROUTE_CHAT_AGENT, with_timing(ROUTE_CHAT_AGENT, chat_agent_node))
builder.add_node(
    ROUTE_NOVEL_SCRIPT_AGENT,
    with_timing(ROUTE_NOVEL_SCRIPT_AGENT, novel_script_agent_node),
)
builder.add_node(NODE_PLANNER, with_timing(NODE_PLANNER, planner_node))
builder.add_node(
    NODE_WORKFLOW_EXECUTOR,
    with_timing(NODE_WORKFLOW_EXECUTOR, workflow_executor_node),
)
builder.add_node(NODE_VERIFIER, with_timing(NODE_VERIFIER, verifier_node))
builder.add_node(NODE_COMPOSER, with_timing(NODE_COMPOSER, composer_node))
builder.add_node(NODE_MERGE, with_timing(NODE_MERGE, merge_node))
builder.add_node(NODE_MEMORY, with_timing(NODE_MEMORY, memory_node))

builder.add_edge(START, NODE_SUPERVISOR)

builder.add_conditional_edges(
    NODE_SUPERVISOR,
    route_after_supervisor,
    {
        ROUTE_TOOL_AGENT: ROUTE_TOOL_AGENT,
        ROUTE_RAG_AGENT: ROUTE_RAG_AGENT,
        ROUTE_CHAT_AGENT: ROUTE_CHAT_AGENT,
        ROUTE_NOVEL_SCRIPT_AGENT: ROUTE_NOVEL_SCRIPT_AGENT,
        NODE_PLANNER: NODE_PLANNER,
    },
)

builder.add_conditional_edges(
    ROUTE_TOOL_AGENT,
    route_after_tool,
    {
        ROUTE_RAG_AGENT: ROUTE_RAG_AGENT,
        NODE_MERGE: NODE_MERGE,
    },
)

# Phase 2：Planner → Executor → Verifier → Composer → memory。
# Planner fail-closed 时 plan 为空，Executor/Verifier/Composer 都会按空 plan 分支
# 给出保守文案，避免把"没生成计划"误报成"执行失败"。
# Composer 自己写 answer，因此 workflow 支路**不再经过 merge_node**（merge 面向
# 多 agent LLM 融合场景，结构化 plan 的合成要确定性输出）。
builder.add_edge(NODE_PLANNER, NODE_WORKFLOW_EXECUTOR)
builder.add_edge(NODE_WORKFLOW_EXECUTOR, NODE_VERIFIER)
builder.add_edge(NODE_VERIFIER, NODE_COMPOSER)
builder.add_edge(NODE_COMPOSER, NODE_MEMORY)

builder.add_edge(ROUTE_RAG_AGENT, NODE_MERGE)
builder.add_edge(ROUTE_CHAT_AGENT, NODE_MERGE)
builder.add_edge(ROUTE_NOVEL_SCRIPT_AGENT, NODE_MERGE)
builder.add_edge(NODE_MERGE, NODE_MEMORY)
builder.add_edge(NODE_MEMORY, END)

graph = builder.compile(checkpointer=get_checkpointer())

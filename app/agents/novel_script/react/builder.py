"""Graph 组装。

单独抽成工厂函数 `build_novel_script_graph` 的原因：
- 测试 / eval 场景里可以拿到一个“干净”的 graph 实例，不用和模块级 singleton 共享状态
- 未来若要按请求传入不同的节点（例如替换 planner_node 为 mock）会更方便

模块级 `novel_script_graph` 仍然保留，保持对 `app.agents.novel_script_agent.py`
等调用点的向后兼容。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agents.novel_script.state import NovelScriptState

from .nodes import (
    finalizer_node,
    planner_node,
    should_continue,
    tool_executor_node,
)


def build_novel_script_graph():
    """构造并编译 novel_script ReAct 图。"""

    builder = StateGraph(NovelScriptState)
    builder.add_node("planner", planner_node)
    builder.add_node("tool_executor", tool_executor_node)
    builder.add_node("finalizer", finalizer_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        should_continue,
        {
            "planner": "tool_executor",
            "finalize": "finalizer",
        },
    )
    builder.add_conditional_edges(
        "tool_executor",
        should_continue,
        {
            "planner": "planner",
            "finalize": "finalizer",
        },
    )
    builder.add_edge("finalizer", END)

    return builder.compile()


# 模块级实例：app.agents.novel_script_agent 直接引用这个对象。
novel_script_graph = build_novel_script_graph()

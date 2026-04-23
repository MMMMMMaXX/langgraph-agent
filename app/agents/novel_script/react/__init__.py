"""novel_script ReAct 子包：planner / tool_executor / finalizer 的具体实现。

对外主要暴露：
- `novel_script_graph` 已编译好的 LangGraph 实例（主链路调用）
- `build_novel_script_graph` 工厂函数（测试 / eval 场景）

其余函数按需从子模块导入。常见入口：
- `from app.agents.novel_script.react import novel_script_graph`
- `from app.agents.novel_script.react.tool_dispatch import TOOLS, TOOL_REDUCERS`
- `from app.agents.novel_script.react.timing import STAGE_PLANNER, add_timing`
"""

from __future__ import annotations

from .builder import build_novel_script_graph, novel_script_graph
from .nodes import (
    finalizer_node,
    planner_node,
    should_continue,
    tool_executor_node,
)
from .planner import (
    build_rule_based_plan,
    has_unreviewed_draft,
    parse_planner_answer,
)
from .timing import add_timing
from .tool_dispatch import (
    TOOL_FACTS,
    TOOL_FINALIZE,
    TOOL_REDUCERS,
    TOOL_REVIEW,
    TOOL_SPLIT,
    TOOL_WRITE,
    TOOLS,
    VALID_SELECTED_TOOLS,
)

__all__ = [
    "build_novel_script_graph",
    "novel_script_graph",
    # nodes
    "planner_node",
    "tool_executor_node",
    "finalizer_node",
    "should_continue",
    # planner helpers
    "build_rule_based_plan",
    "has_unreviewed_draft",
    "parse_planner_answer",
    # timing
    "add_timing",
    # tool dispatch
    "TOOL_SPLIT",
    "TOOL_FACTS",
    "TOOL_WRITE",
    "TOOL_REVIEW",
    "TOOL_FINALIZE",
    "TOOLS",
    "TOOL_REDUCERS",
    "VALID_SELECTED_TOOLS",
]

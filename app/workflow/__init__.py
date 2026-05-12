"""Workflow 相关模块：Plan schema + tool registry（业务名 ↔ function name）。

对外暴露：
- `WorkflowStep / Plan`：Planner 产出 / Executor 消费的结构化数据。
- `ToolRegistry`：维护业务名（带点号）与 function-calling name（下划线）的双向映射，
  并按 `AuthContext` 裁剪可见工具。
- `parse_plan`：把 LLM 输出的 JSON 字符串校验为 `Plan` 对象，失败抛 `PlanValidationError`。
"""

from __future__ import annotations

from app.workflow.registry import ToolRegistry, default_tool_registry
from app.workflow.schema import (
    Plan,
    PlanValidationError,
    WorkflowStep,
    parse_plan,
)

__all__ = [
    "Plan",
    "PlanValidationError",
    "ToolRegistry",
    "WorkflowStep",
    "default_tool_registry",
    "parse_plan",
]

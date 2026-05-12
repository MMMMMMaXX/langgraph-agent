"""Tool Registry：Planner 业务名 ↔ function-calling name 的解析层。

背景：
- OpenAI function-calling 要求 name 匹配 `^[a-zA-Z0-9_-]+$`，所以底层 tool spec
  一律用下划线 name（如 `ticket_create`）。
- Planner prompt 面向 LLM 时倾向于暴露"业务风格"名字（点号分段，如
  `ticket.create`、`monitor.query_errors`），让 LLM 更容易生成合理 plan。

解析规则（单向收敛到 canonical function name）：
1. 如果 `name` 已经在 `TOOL_METADATA` 中 → 就是 canonical。
2. 否则把所有 `.` 替换为 `_` 再查一次 → 命中则视为业务别名。
3. 否则抛 `ToolNotRegisteredError`。

这样 Planner 写 `ticket.create` / `ticket_create` / 甚至 `foo.bar.ticket_create`
都会落到同一个 canonical function name，避免 Executor 这层还要再处理命名差异。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.auth.context import AuthContext
from app.tools.metadata import (
    TOOL_METADATA,
    ToolMetadata,
    ToolNotRegisteredError,
    get_tool_metadata,
)


@dataclass(frozen=True)
class ToolRegistry:
    """封装 TOOL_METADATA 的视图，提供 Planner/Verifier 需要的查询方法。

    之所以是 frozen dataclass 而非模块级函数集合：让测试可以注入一个裁剪后的
    registry（例如只开放 `get_weather`），而不必 monkeypatch 全局 TOOL_METADATA。
    """

    metadata: dict[str, ToolMetadata]

    # ---- 基础解析 ---------------------------------------------------------

    def resolve(self, name: str) -> str:
        """把 Planner 给的名字收敛到 canonical function name。

        任何一个点号都会被替换为下划线；未登记时抛 ToolNotRegisteredError。
        """

        if not name:
            raise ToolNotRegisteredError(name)
        if name in self.metadata:
            return name
        underscored = name.replace(".", "_")
        if underscored in self.metadata:
            return underscored
        raise ToolNotRegisteredError(name)

    def get(self, name: str) -> ToolMetadata:
        """按业务名或 function name 取 metadata；未登记抛。"""

        return self.metadata[self.resolve(name)]

    def has(self, name: str) -> bool:
        try:
            self.resolve(name)
            return True
        except ToolNotRegisteredError:
            return False

    # ---- Planner prompt 构造 ---------------------------------------------

    def visible_tools(self, auth: AuthContext) -> tuple[ToolMetadata, ...]:
        """按 AuthContext 裁剪出 Planner 应该看见的 tool 列表。

        复用 `filter_tools_for_auth` 的策略：匿名用户看不到 side_effect 工具。
        Planner prompt 生成时调这个方法，防止 LLM 产出越权 plan。
        """

        return tuple(
            meta
            for meta in self.metadata.values()
            if not (auth.anonymous and meta.side_effect)
        )

    def as_function_names(self, metas: Iterable[ToolMetadata]) -> tuple[str, ...]:
        """方便 Planner prompt 列出允许的 function name 清单。"""

        return tuple(meta.name for meta in metas)


default_tool_registry: ToolRegistry = ToolRegistry(metadata=TOOL_METADATA)
"""进程级默认 registry，直接复用全局 TOOL_METADATA。

测试需要裁剪时：`ToolRegistry(metadata={...})` 自己造一个，不要 patch 全局。
"""


__all__ = [
    "ToolRegistry",
    "default_tool_registry",
    # 透出便于调用方直接 except，不用再 import metadata
    "ToolNotRegisteredError",
    "get_tool_metadata",
]

"""Tool Metadata Registry：集中声明每个工具的安全属性。

关键不变量（`ToolMetadata.__post_init__` 校验）：
- `read_only` 和 `side_effect` **互斥**：工具要么纯读，要么有副作用，不允许
  既声明 read_only 又声明 side_effect。
- read_only 工具禁止要求 confirmation / idempotency：省掉流程但也明确责任边界。
- side_effect 工具**必须**同时要求 confirmation + idempotency：这是 Phase 1
  的硬约束，避免将来加了副作用工具却忘记任一条兜底。

`filter_tools_for_auth` 做身份 × metadata 的策略执行：
- 匿名上下文（`auth.anonymous=True`）一律看不到 side_effect 工具；
- tools 列表里出现未登记的工具 → 立刻抛，防止 LLM function-calling schema
  和 metadata registry 不同步。

对外导出：`ToolMetadata` / `TOOL_METADATA` / `get_tool_metadata` /
`filter_tools_for_auth`。tool_agent 在调 `chat_with_tools` 前调用后两者。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from app.constants.tool_safety import (
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    ERR_TOOL_NOT_REGISTERED,
    RISK_LEVEL_LOW,
    RISK_LEVEL_MEDIUM,
    VALID_RISK_LEVELS,
)
from app.constants.tooling import (
    TOOL_NAME_CALCULATE,
    TOOL_NAME_GET_WEATHER,
    TOOL_NAME_MONITOR_QUERY_ERRORS,
    TOOL_NAME_TICKET_CREATE,
)


@dataclass(frozen=True)
class ToolMetadata:
    """单个工具的安全声明。"""

    name: str
    read_only: bool
    side_effect: bool
    requires_confirmation: bool
    idempotency_required: bool
    risk_level: str
    timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if self.risk_level not in VALID_RISK_LEVELS:
            raise ValueError(f"invalid risk level: {self.risk_level!r}")

        # read_only / side_effect 互斥。
        if self.read_only == self.side_effect:
            raise ValueError(
                f"tool {self.name!r}: read_only and side_effect must be "
                "mutually exclusive"
            )
        if self.read_only and (self.requires_confirmation or self.idempotency_required):
            raise ValueError(
                f"tool {self.name!r}: read_only tools must not require "
                "confirmation or idempotency"
            )
        if self.side_effect and not (
            self.requires_confirmation and self.idempotency_required
        ):
            raise ValueError(
                f"tool {self.name!r}: side_effect tools must require both "
                "confirmation and idempotency"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(f"tool {self.name!r}: timeout_seconds must be positive")


# 当前所有在册工具。Phase 1 两个都是 read_only；PR 4 会加上 ticket_create。
TOOL_METADATA: dict[str, ToolMetadata] = {
    TOOL_NAME_GET_WEATHER: ToolMetadata(
        name=TOOL_NAME_GET_WEATHER,
        read_only=True,
        side_effect=False,
        requires_confirmation=False,
        idempotency_required=False,
        risk_level=RISK_LEVEL_LOW,
    ),
    TOOL_NAME_CALCULATE: ToolMetadata(
        name=TOOL_NAME_CALCULATE,
        read_only=True,
        side_effect=False,
        requires_confirmation=False,
        idempotency_required=False,
        risk_level=RISK_LEVEL_LOW,
    ),
    TOOL_NAME_TICKET_CREATE: ToolMetadata(
        name=TOOL_NAME_TICKET_CREATE,
        read_only=False,
        side_effect=True,
        requires_confirmation=True,
        idempotency_required=True,
        risk_level=RISK_LEVEL_MEDIUM,
    ),
    TOOL_NAME_MONITOR_QUERY_ERRORS: ToolMetadata(
        name=TOOL_NAME_MONITOR_QUERY_ERRORS,
        read_only=True,
        side_effect=False,
        requires_confirmation=False,
        idempotency_required=False,
        risk_level=RISK_LEVEL_LOW,
    ),
}


class ToolNotRegisteredError(KeyError):
    """工具名未在 TOOL_METADATA 登记。

    之所以继承 KeyError：下游常见写法 `TOOL_METADATA[name]` 捕获的是 KeyError，
    这里保持向下兼容，同时通过类型又能让上层按业务语义特判。
    """

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"{ERR_TOOL_NOT_REGISTERED}: {tool_name}")
        self.tool_name = tool_name


def get_tool_metadata(name: str) -> ToolMetadata:
    """按名取 metadata；未登记则抛 ToolNotRegisteredError。"""

    try:
        return TOOL_METADATA[name]
    except KeyError as exc:
        raise ToolNotRegisteredError(name) from exc


def _tool_name_of(spec: dict[str, Any]) -> str:
    """从 OpenAI function-calling tool spec 里取工具名。

    Spec 结构：`{"type": "function", "function": {"name": "...", ...}}`。
    任何一层缺失都视为结构错误，立即抛。
    """

    try:
        return spec["function"]["name"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid tool spec: {spec!r}") from exc


def filter_tools_for_auth(
    tools: list[dict[str, Any]],
    tool_impls: Mapping[str, Callable[..., Any]],
    *,
    anonymous: bool,
) -> tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]:
    """根据 AuthContext.anonymous 过滤工具白名单。

    - 匿名 × side_effect：工具直接从 LLM 可见列表移除，也不注入 impl；
      Phase 1 文档要求这种组合在 API 层 / agent 层双重拦截，这里是 agent 层。
    - 所有 tools 必须已登记，否则抛 ToolNotRegisteredError（启动期保证一致性）。

    返回 `(filtered_tools, filtered_impls)`，两者保持 name 一一对应。
    """

    filtered_tools: list[dict[str, Any]] = []
    filtered_impls: dict[str, Callable[..., Any]] = {}

    for spec in tools:
        name = _tool_name_of(spec)
        meta = get_tool_metadata(name)

        if anonymous and meta.side_effect:
            # 安全静默：不给 LLM 看见，也就不会触发 tool_call。
            continue

        filtered_tools.append(spec)
        if name in tool_impls:
            filtered_impls[name] = tool_impls[name]

    return filtered_tools, filtered_impls


__all__ = [
    "TOOL_METADATA",
    "ToolMetadata",
    "ToolNotRegisteredError",
    "filter_tools_for_auth",
    "get_tool_metadata",
]

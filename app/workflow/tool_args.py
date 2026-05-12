"""Tool args 单一事实源：从 OpenAI function-calling specs 派生各类索引。

背景：
- Planner 输出是 free-form JSON，没有协议层 schema 强校验，LLM 会幻觉出
  spec 里未声明的参数（例如给 monitor_query_errors 塞 `level` / `time_range`）。
- Executor 需要一份"允许参数"白名单做过滤；Verifier 需要"必需参数"清单做
  完整性校验；Planner prompt 需要"参数摘要（含是否必填）"用来约束 LLM 生成。
- 这三份信息本质都来自同一份 OpenAI function-calling spec
  （`app.agents.tool_agent.TOOLS`）。历史上 Executor 和 Verifier 各写了一份
  几乎一样的 builder，违反常量抽离原则；本模块收拢到一处，新增工具只改 spec。

对外导出：
- `build_*_index`：纯函数 builder，便于单测注入伪造 specs。
- `ALLOWED_ARGS / REQUIRED_ARGS / ARG_SUMMARIES`：模块级缓存，调用方直接读。
- `filter_args_by_spec`：Executor 的参数过滤工具函数。
- `ToolArgSummary`：Planner prompt 渲染用的结构化参数描述。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.tool_agent import TOOLS as _TOOL_SPECS


@dataclass(frozen=True)
class ToolArgSummary:
    """单个参数的可读摘要。

    用于 Planner prompt 渲染；保持字段稳定，测试可以直接比较。
    """

    name: str
    required: bool
    type: str
    description: str


def _iter_spec_params(
    specs: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any], tuple[str, ...]]]:
    """把 specs 拆成 `[(tool_name, properties, required_tuple), ...]`。

    私有辅助：各 builder 共用的遍历逻辑，避免三份 builder 各自 try/except。
    非法结构（缺 function/name 等）在这里直接跳过；spec 是内部维护的常量，
    出错属于编码问题，尽早报错会由下游（例如 function-calling 调用）抛出。
    """

    out: list[tuple[str, dict[str, Any], tuple[str, ...]]] = []
    for spec in specs:
        fn = spec.get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        required = tuple(params.get("required") or ())
        if not isinstance(props, dict):
            continue
        out.append((name, props, required))
    return out


def build_allowed_args_index(
    specs: list[dict[str, Any]],
) -> dict[str, frozenset[str]]:
    """派生 `{tool_name: frozenset(allowed_arg_keys)}`。

    Executor 按此白名单过滤 Planner 产出，能把 LLM 幻觉从"工具崩溃"降级成
    "参数被静默丢弃 + 一条 WARNING"。
    """

    return {name: frozenset(props.keys()) for name, props, _ in _iter_spec_params(specs)}


def build_required_args_index(
    specs: list[dict[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """派生 `{tool_name: (required_arg,...)}`，供 Verifier 检查参数完整性。"""

    return {name: required for name, _, required in _iter_spec_params(specs)}


def build_arg_summaries(
    specs: list[dict[str, Any]],
) -> dict[str, tuple[ToolArgSummary, ...]]:
    """派生 `{tool_name: (ToolArgSummary, ...)}`，供 Planner prompt 渲染。

    顺序：先按 `required` 列表排（保留 spec 作者意图的优先级），再按
    剩余 key 的字典序。这样 prompt 输出稳定，不会因 dict 遍历顺序抖动。
    """

    out: dict[str, tuple[ToolArgSummary, ...]] = {}
    for name, props, required in _iter_spec_params(specs):
        required_set = set(required)
        ordered_keys: list[str] = list(required)
        ordered_keys.extend(sorted(k for k in props.keys() if k not in required_set))
        summaries: list[ToolArgSummary] = []
        for key in ordered_keys:
            schema = props.get(key) or {}
            summaries.append(
                ToolArgSummary(
                    name=key,
                    required=key in required_set,
                    type=str(schema.get("type") or "any"),
                    description=str(schema.get("description") or "").strip(),
                )
            )
        out[name] = tuple(summaries)
    return out


# 模块级派生常量。调用方（executor / verifier / prompt builder）直接读即可。
ALLOWED_ARGS: dict[str, frozenset[str]] = build_allowed_args_index(_TOOL_SPECS)
REQUIRED_ARGS: dict[str, tuple[str, ...]] = build_required_args_index(_TOOL_SPECS)
ARG_SUMMARIES: dict[str, tuple[ToolArgSummary, ...]] = build_arg_summaries(_TOOL_SPECS)


def filter_args_by_spec(
    canonical: str, args: dict[str, Any]
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """按 spec 白名单过滤 args，返回 `(filtered_args, dropped_keys)`。

    未登记的工具保持原样（让后续 `impl(**args)` 报错，能在日志里看到真实错误）。
    """

    allowed = ALLOWED_ARGS.get(canonical)
    if allowed is None:
        return args, ()
    filtered = {k: v for k, v in args.items() if k in allowed}
    dropped = tuple(k for k in args.keys() if k not in allowed)
    return filtered, dropped


__all__ = [
    "ALLOWED_ARGS",
    "ARG_SUMMARIES",
    "REQUIRED_ARGS",
    "ToolArgSummary",
    "build_allowed_args_index",
    "build_arg_summaries",
    "build_required_args_index",
    "filter_args_by_spec",
]

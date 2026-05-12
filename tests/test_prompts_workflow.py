"""Unit tests: app/prompts/workflow.py (Planner prompt rendering)."""

from __future__ import annotations

from app.prompts.workflow import build_planner_system_prompt
from app.tools.metadata import TOOL_METADATA


def _meta(name: str):
    return TOOL_METADATA[name]


def test_prompt_lists_tool_args_with_required_marker() -> None:
    # monitor_query_errors 只有一个 required 字段 service；prompt 必须显式列出，
    # 并带 `*` 标记。Planner 原本会幻觉 time_range/level，rule 9 + 参数白名单
    # 共同压制这个幻觉面。
    prompt = build_planner_system_prompt((_meta("monitor_query_errors"),))
    assert "monitor.query.errors" in prompt
    assert "function name: monitor_query_errors" in prompt
    assert "args:" in prompt
    assert "service*:string" in prompt
    # 描述里的关键词应当被带进来，帮 Planner 选对 service 名。
    assert "payment-service" in prompt


def test_prompt_rule_9_forbids_inventing_arg_keys() -> None:
    # 硬性规则 9 显式禁止幻觉字段；这些关键词出现在规则文本中是这项防御的可测面。
    prompt = build_planner_system_prompt((_meta("get_weather"),))
    assert "args 的键名必须严格" in prompt
    # 列举一部分历史幻觉样本，确保规则里有引导。
    assert "time_range" in prompt or "error_type" in prompt


def test_prompt_empty_visible_tools_falls_back_to_placeholder() -> None:
    prompt = build_planner_system_prompt(())
    # 没有可用工具时仍然输出一行占位，避免 LLM 因为空列表自由发挥。
    assert "当前身份下没有可用工具" in prompt

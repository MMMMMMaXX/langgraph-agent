"""Unit tests: app/workflow/tool_args.py."""

from __future__ import annotations

from app.workflow.tool_args import (
    ALLOWED_ARGS,
    ARG_SUMMARIES,
    REQUIRED_ARGS,
    ToolArgSummary,
    build_allowed_args_index,
    build_arg_summaries,
    build_required_args_index,
    filter_args_by_spec,
)

# 精心构造的测试 specs：覆盖有 required / 无 required / 参数顺序稳定性。
_SAMPLE_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "alpha",
            "description": "alpha desc",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "string", "description": "A 字段"},
                    "b": {"type": "integer"},  # 无描述，仅 type
                },
                "required": ["a"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "beta",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    # 脏数据：缺 name 的 spec 要被忽略，不能炸。
    {"type": "function", "function": {"parameters": {"properties": {"x": {}}}}},
]


class TestBuilders:
    def test_allowed_args_index_uses_properties(self) -> None:
        idx = build_allowed_args_index(_SAMPLE_SPECS)
        assert idx["alpha"] == frozenset({"a", "b"})
        assert idx["beta"] == frozenset()
        assert "(" not in " ".join(idx)  # 非法 spec 被丢弃

    def test_required_args_index_preserves_order(self) -> None:
        # spec 里 required=["a"]，派生结果保留 tuple 顺序。
        idx = build_required_args_index(_SAMPLE_SPECS)
        assert idx["alpha"] == ("a",)
        assert idx["beta"] == ()

    def test_arg_summaries_required_first_then_alpha(self) -> None:
        # 优先按 required 声明顺序，再对剩余字典序排序；保证 prompt 输出稳定。
        summaries = build_arg_summaries(_SAMPLE_SPECS)["alpha"]
        assert summaries == (
            ToolArgSummary(
                name="a", required=True, type="string", description="A 字段"
            ),
            ToolArgSummary(name="b", required=False, type="integer", description=""),
        )

    def test_arg_summaries_empty_tool(self) -> None:
        assert build_arg_summaries(_SAMPLE_SPECS)["beta"] == ()


class TestModuleLevelConstants:
    def test_derived_from_real_tool_specs(self) -> None:
        # monitor_query_errors 只有一个 required 字段 service；
        # 这是 "新工具加入 / 参数调整" 时最有可能漏改的地方，用真实 spec 校验。
        assert ALLOWED_ARGS["monitor_query_errors"] == frozenset({"service"})
        assert REQUIRED_ARGS["monitor_query_errors"] == ("service",)
        summaries = ARG_SUMMARIES["monitor_query_errors"]
        assert len(summaries) == 1
        assert summaries[0].name == "service"
        assert summaries[0].required is True
        assert summaries[0].type == "string"
        assert "payment-service" in summaries[0].description


class TestFilterArgsBySpec:
    def test_drops_unknown_keys(self) -> None:
        filtered, dropped = filter_args_by_spec(
            "monitor_query_errors",
            {"service": "payment-service", "level": "error", "time_range": "30m"},
        )
        assert filtered == {"service": "payment-service"}
        # 按原字典顺序返回，调用方记日志用。
        assert dropped == ("level", "time_range")

    def test_unknown_tool_passthrough(self) -> None:
        # 未登记的工具保持原样；让后续 impl(**args) 抛出真实错误便于定位。
        args = {"anything": 1}
        filtered, dropped = filter_args_by_spec("tool_not_in_specs", args)
        assert filtered is args
        assert dropped == ()

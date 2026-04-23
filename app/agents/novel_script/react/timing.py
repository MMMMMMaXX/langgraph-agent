"""分阶段耗时打点工具。

novel_script 是一个迭代式 ReAct agent，单一的“总耗时”无法定位瓶颈。
这里做两件事：
1. 维护 `timing_breakdown_ms`：按 stage 聚合累计耗时（planner / write_script_scene 等）
2. 维护 `iteration_timings`：逐轮明细，带上 iteration 序号和 extra 字段

把 stage 名称集中成 `STAGE_*` 常量可以避免节点里散落魔法字符串，
打点/查询/dashboard 渲染都只会引用到同一组常量。
"""

from __future__ import annotations

from app.agents.novel_script.state import NovelScriptState

# ---- planner 阶段细分 ----
STAGE_PLANNER = "planner"
STAGE_PLANNER_CONTEXT = "planner_context_build"
STAGE_PLANNER_LLM = "planner_llm_call"
STAGE_PLANNER_PARSE = "planner_parse"
STAGE_PLANNER_FALLBACK = "planner_fallback_plan"

# ---- tool executor 阶段 ----
# 具体 tool 的 stage 名就是 tool_name 本身（split_into_scenes / write_script_scene 等），
# 这样 timing 报表可以直接看到“哪个工具累计耗时最高”，不需要再做名字映射。
STAGE_TOOL_FINALIZE = "tool_executor_finalize"

# ---- finalizer ----
STAGE_FINALIZER = "finalizer"


def add_timing(
    state: NovelScriptState,
    key: str,
    duration_ms: float,
    extra: dict | None = None,
) -> NovelScriptState:
    """追加一次阶段耗时记录。

    约定：iteration_timings 每条的 `iteration` 字段取自 `state["iteration_count"]`。
    planner_node 在进入 LLM 调用前会先递增 iteration_count，因此 planner_* 阶段
    打出来的 iteration 是“当前正在执行的这一轮”，tool_executor 阶段打出来的
    iteration 与其对应。调用方在改顺序时需要留意这个约定。

    creative agent 的性能问题通常不是“整条链路慢”，而是某一轮某一步特别慢。
    这里同时记录：
    1. 总体分阶段累计耗时（timing_breakdown_ms）
    2. 每一轮的明细耗时（iteration_timings）
    这样日志里能直接回答“卡在第几轮、哪一步”。
    """

    timing_breakdown_ms = dict(state.get("timing_breakdown_ms", {}))
    timing_breakdown_ms[key] = round(
        timing_breakdown_ms.get(key, 0.0) + duration_ms, 2
    )

    iteration_timings = list(state.get("iteration_timings", []))
    item: dict = {
        "iteration": state.get("iteration_count", 0),
        "stage": key,
        "duration_ms": round(duration_ms, 2),
    }
    if extra:
        item.update(extra)
    iteration_timings.append(item)

    return {
        **state,
        "timing_breakdown_ms": timing_breakdown_ms,
        "iteration_timings": iteration_timings,
    }

"""LangGraph 节点实现。

三个节点 + 一个路由函数：
- planner_node      调用 LLM planner（失败时回落到规则 planner）
- tool_executor_node 执行 planner 选择的工具，把结果 merge 回 state
- finalizer_node    确定性组装最终剧本
- should_continue   planner/tool_executor 后判断下一跳

这里尽量只做“协调 + 打点”，具体逻辑都被拆到同级模块：
- planner / context / review 负责“想什么”
- tool_dispatch / finalizer 负责“做什么”
- timing 管 stage 常量和打点记录
"""

from __future__ import annotations

from app.agents.novel_script.constants import (
    OBSERVATION_PREVIEW_CHARS,
    PLANNER_THOUGHT_PREVIEW_CHARS,
    TOOL_OUTPUT_PREVIEW_CHARS,
)
from app.agents.novel_script.prompts import REACT_PLANNER_PROMPT
from app.agents.novel_script.state import NovelScriptState
from app.constants.model_profiles import PROFILE_CREATIVE_PLANNER
from app.llm import chat, get_profile_runtime_info
from app.utils.errors import build_error_info
from app.utils.logger import now_ms

from .context import build_planner_context
from .finalizer import build_deterministic_final_script
from .planner import build_rule_based_plan, has_unreviewed_draft, parse_planner_answer
from .timing import (
    STAGE_FINALIZER,
    STAGE_PLANNER,
    STAGE_PLANNER_CONTEXT,
    STAGE_PLANNER_FALLBACK,
    STAGE_PLANNER_LLM,
    STAGE_PLANNER_PARSE,
    STAGE_TOOL_FINALIZE,
    add_timing,
)
from .tool_dispatch import (
    TOOL_FINALIZE,
    TOOL_REDUCERS,
    TOOLS,
    filter_tool_input,
    get_tool_llm_profile,
    normalize_tool_input,
)


def planner_node(state: NovelScriptState) -> NovelScriptState:
    """ReAct graph 中的“思考”节点。

    - 调用 LLM planner 决定下一步动作
    - 解析失败/超时时自动切到 rule-based fallback
    - 每个子阶段（构造上下文 / LLM 调用 / 解析 / fallback）单独打点
      便于后续定位“到底慢在哪一段”
    """

    started_at_ms = now_ms()
    planner_error = ""
    planner_mode = "react_llm"
    planner_answer = ""
    planner_llm_info = get_profile_runtime_info(PROFILE_CREATIVE_PLANNER)

    # --- 阶段 1：上下文构造 ---
    context_started_at_ms = now_ms()
    planner_context = build_planner_context(state)
    context_duration_ms = now_ms() - context_started_at_ms
    timing_state = add_timing(
        state,
        STAGE_PLANNER_CONTEXT,
        context_duration_ms,
        extra={
            "context_chars": len(planner_context),
            "scene_count": len(state.get("scene_plan", [])),
            "draft_count": len(state.get("scene_drafts", [])),
            "review_count": len(state.get("review_notes", [])),
            **planner_llm_info,
        },
    )

    # --- 阶段 2：LLM 调用 + 解析 ---
    llm_started_at_ms = now_ms()
    try:
        planner_answer = chat(
            [
                {"role": "system", "content": REACT_PLANNER_PROMPT},
                {"role": "user", "content": planner_context},
            ],
            max_completion_tokens=220,
            profile=PROFILE_CREATIVE_PLANNER,
        )
        llm_duration_ms = now_ms() - llm_started_at_ms
        timing_state = add_timing(
            timing_state,
            STAGE_PLANNER_LLM,
            llm_duration_ms,
            extra={
                "context_chars": len(planner_context),
                "answer_chars": len(planner_answer),
                **planner_llm_info,
            },
        )

        parse_started_at_ms = now_ms()
        plan = parse_planner_answer(planner_answer)
        parse_duration_ms = now_ms() - parse_started_at_ms
        timing_state = add_timing(
            timing_state,
            STAGE_PLANNER_PARSE,
            parse_duration_ms,
            extra={
                "selected_tool": (plan or {}).get("selected_tool", ""),
                **planner_llm_info,
            },
        )
    except Exception as exc:
        llm_duration_ms = now_ms() - llm_started_at_ms
        timing_state = add_timing(
            timing_state,
            STAGE_PLANNER_LLM,
            llm_duration_ms,
            extra={
                "context_chars": len(planner_context),
                "answer_chars": len(planner_answer),
                "error": str(exc),
                **planner_llm_info,
            },
        )
        plan = None
        planner_error = build_error_info(exc, stage="planner", source="llm")
        planner_mode = "react_fallback_rule"

    # --- 阶段 3：fallback ---
    # 当模型超时、限流、输出坏 JSON 或工具名非法时，都回退到规则计划器，
    # 保证链路继续执行。
    if plan is None:
        fallback_started_at_ms = now_ms()
        plan = build_rule_based_plan(timing_state)
        fallback_duration_ms = now_ms() - fallback_started_at_ms
        timing_state = add_timing(
            timing_state,
            STAGE_PLANNER_FALLBACK,
            fallback_duration_ms,
            extra={
                "selected_tool": plan.get("selected_tool", TOOL_FINALIZE),
                "reason": planner_error or "invalid planner output",
            },
        )
        if not planner_error:
            planner_mode = "react_fallback_rule"

    next_state: NovelScriptState = {
        **timing_state,
        "thought": plan.get("thought", ""),
        "selected_tool": plan.get("selected_tool", TOOL_FINALIZE),
        "tool_input": plan.get("tool_input", {}),
        "iteration_count": state.get("iteration_count", 0) + 1,
    }
    return add_timing(
        next_state,
        STAGE_PLANNER,
        now_ms() - started_at_ms,
        extra={
            "mode": planner_mode,
            "selected_tool": next_state.get("selected_tool", TOOL_FINALIZE),
            "thought": next_state.get("thought", "")[:PLANNER_THOUGHT_PREVIEW_CHARS],
            "context_chars": len(planner_context),
            "planner_answer_chars": len(planner_answer),
            **planner_llm_info,
            "error": planner_error,
        },
    )


def tool_executor_node(state: NovelScriptState) -> NovelScriptState:
    """ReAct graph 中的“行动”节点。

    1. 读取 planner 刚才选择的工具
    2. 补全参数并过滤非法字段
    3. 调用对应 Python 工具函数
    4. 通过 TOOL_REDUCERS 把结果 merge 回 state
    5. 根据 iteration/max/未审查草稿决定是否标记 done

    状态合并逻辑集中在 tool_dispatch.apply_*_result，
    本节点主要负责执行流程 + timing + 错误兜底。
    """

    started_at_ms = now_ms()
    tool_name = state.get("selected_tool", TOOL_FINALIZE)
    tool_input = normalize_tool_input(
        tool_name,
        dict(state.get("tool_input", {})),
        state,
    )
    history = list(state.get("tool_history", []))

    # --- 终态 ---
    if tool_name == TOOL_FINALIZE:
        next_state = {**state, "done": True}
        return add_timing(next_state, STAGE_TOOL_FINALIZE, now_ms() - started_at_ms)

    tool_input = filter_tool_input(tool_name, tool_input)
    tool_fn = TOOLS[tool_name]
    tool_profile = get_tool_llm_profile(tool_name)
    tool_llm_info = get_profile_runtime_info(tool_profile) if tool_profile else {}

    # --- 执行工具 ---
    try:
        result = tool_fn(**tool_input)
    except Exception as exc:
        error_info = build_error_info(
            exc,
            stage=tool_name,
            source="novel_script_tool",
            preferred_code="tool_execution_error",
        )
        history.append(
            {
                "thought": state.get("thought", ""),
                "selected_tool": tool_name,
                "tool_input": tool_input,
                "tool_error": error_info,
            }
        )
        next_state: NovelScriptState = {
            **state,
            "observation": f"{tool_name} 执行失败：{error_info['message']}",
            "tool_history": history,
            "done": True,
        }
        return add_timing(
            next_state,
            tool_name,
            now_ms() - started_at_ms,
            extra={
                "selected_tool": tool_name,
                "scene_count": len(next_state.get("scene_plan", [])),
                "draft_count": len(next_state.get("scene_drafts", [])),
                "review_count": len(next_state.get("review_notes", [])),
                **tool_llm_info,
                "error": error_info,
            },
        )

    # --- 工具成功：记录 history + 应用 reducer ---
    history.append(
        {
            "thought": state.get("thought", ""),
            "selected_tool": tool_name,
            "tool_input": tool_input,
            "tool_output_preview": str(result)[:TOOL_OUTPUT_PREVIEW_CHARS],
        }
    )

    next_state: NovelScriptState = {
        **state,
        "tool_output": result,
        "observation": str(result)[:OBSERVATION_PREVIEW_CHARS],
        "tool_history": history,
    }

    reducer = TOOL_REDUCERS.get(tool_name)
    if reducer is not None:
        next_state = {**next_state, **reducer(result, state)}

    # --- 迭代上限兜底：达到 max 时若没有未审查草稿就结束 ---
    if next_state.get("iteration_count", 0) >= next_state.get(
        "max_iterations", 6
    ) and not has_unreviewed_draft(next_state):
        next_state["done"] = True

    return add_timing(
        next_state,
        tool_name,
        now_ms() - started_at_ms,
        extra={
            "selected_tool": tool_name,
            "scene_count": len(next_state.get("scene_plan", [])),
            "draft_count": len(next_state.get("scene_drafts", [])),
            "review_count": len(next_state.get("review_notes", [])),
            **tool_llm_info,
        },
    )


def should_continue(state: NovelScriptState) -> str:
    """planner / tool_executor 出口路由。"""

    if state.get("done"):
        return "finalize"
    if state.get("selected_tool") == TOOL_FINALIZE:
        return "finalize"
    return "planner"


def finalizer_node(state: NovelScriptState) -> NovelScriptState:
    """ReAct graph 的收尾节点。

    当前 finalizer 是纯确定性组装器，不再调用 LLM。
    只负责：
    - 从 state 中读取 scene_drafts / facts / review 等中间结果
    - 生成最终输出字符串
    - 标记 done=True
    """

    started_at_ms = now_ms()
    finalizer_strategy = "deterministic_assembler"
    final_script = build_deterministic_final_script(state)

    next_state: NovelScriptState = {
        **state,
        "final_script": final_script,
        "done": True,
    }
    return add_timing(
        next_state,
        STAGE_FINALIZER,
        now_ms() - started_at_ms,
        extra={
            "strategy": finalizer_strategy,
            "answer_chars": len(final_script),
        },
    )

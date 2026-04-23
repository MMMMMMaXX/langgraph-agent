"""Planner 决策层：LLM planner + 规则 fallback + 相关判定。

ReAct agent 的 planner 负责“想下一步做什么”：
- 优先走 LLM planner：输出 JSON，解析出 thought / selected_tool / tool_input
- 模型异常、超时、输出非法 JSON 或工具名越界时，回落到 rule-based planner
- rule-based planner 不调 LLM，按“拆场景 → 抽事实 → 写场景 → review → 重写 → finalize”
  这条稳定路径兜底，保证链路总能走完

节点里（nodes.py）只负责串起“调用 planner、打 timing、更新 state”；
本文件只产出“下一步是什么”，不读写 timing、不直接依赖 LangGraph。
"""

from __future__ import annotations

import json

from app.agents.novel_script.constants import DEFAULT_TARGET_SCENE_COUNT
from app.agents.novel_script.state import NovelScriptState

from .review import collect_review_issues, scene_has_rewrite_budget
from .tool_dispatch import (
    TOOL_FACTS,
    TOOL_FINALIZE,
    TOOL_REVIEW,
    TOOL_SPLIT,
    TOOL_WRITE,
    VALID_SELECTED_TOOLS,
    find_scene_plan_item,
    normalize_tool_input,
)


def parse_planner_answer(answer: str) -> dict | None:
    """解析 planner 输出的 JSON，并做最基础的合法性校验。

    校验项只有两类：
    1. JSON 是否可解析
    2. selected_tool 是否属于允许的工具集合（含 finalize）

    不满足则返回 None，让上层回退到 rule-based planner。
    """

    try:
        parsed = json.loads(answer)
    except Exception:
        return None

    selected_tool = parsed.get("selected_tool")
    if selected_tool not in VALID_SELECTED_TOOLS:
        return None

    tool_input = parsed.get("tool_input", {})
    if not isinstance(tool_input, dict):
        tool_input = {}

    return {
        "thought": str(parsed.get("thought", "")).strip(),
        "selected_tool": selected_tool,
        "tool_input": tool_input,
    }


def has_unreviewed_draft(state: NovelScriptState) -> bool:
    """判断当前最新版草稿是否还没经过 review。

    用于迭代上限兜底：rewrite 刚刚改动场景后 draft_version 会递增，
    此时即使接近 max_iterations 也应该先允许一次复审，
    否则“改了但没验收”的问题会被吞掉。
    """

    if not state.get("enable_review", True):
        return False
    if not state.get("scene_drafts"):
        return False
    return state.get("last_reviewed_draft_version", -1) < state.get(
        "draft_version", 0
    )


def _decide_rule_based_action(
    state: NovelScriptState,
) -> tuple[str, str, dict]:
    """规则 planner 的决策核心。

    返回 (tool_name, thought, overrides)：
    - tool_name：下一步工具名（含 TOOL_FINALIZE）
    - thought：写给日志/前端的一句中文解释
    - overrides：要传给 normalize_tool_input 的显式字段（例如重写时的 scene_id）

    把“决策”从“参数构造”里解耦，是为了让 rule planner 不再和
    normalize_tool_input 各写一份相同的“从 scene_plan 取下一个场景”逻辑。
    """

    scene_plan = state.get("scene_plan", [])
    story_facts = state.get("story_facts", {})
    scene_drafts = state.get("scene_drafts", [])
    review_notes = state.get("review_notes", [])
    pending_rewrite_scene_ids = list(state.get("pending_rewrite_scene_ids", []))
    enable_review = state.get("enable_review", True)
    draft_version = state.get("draft_version", 0)
    last_reviewed_draft_version = state.get("last_reviewed_draft_version", -1)

    if not scene_plan:
        return (
            TOOL_SPLIT,
            "先拆分场景，建立改编骨架。",
            {
                "text": state.get("source_text", ""),
                "target_scene_count": state.get(
                    "target_scene_count", DEFAULT_TARGET_SCENE_COUNT
                ),
            },
        )

    if not story_facts:
        return (
            TOOL_FACTS,
            "先抽取故事事实，避免后续改编偏离原文。",
            {"text": state.get("source_text", "")},
        )

    if len(scene_drafts) < len(scene_plan):
        next_scene = scene_plan[len(scene_drafts)]
        scene_id = next_scene.get("scene_id", f"scene_{len(scene_drafts) + 1}")
        return (
            TOOL_WRITE,
            f"开始生成 {scene_id} 的剧本草稿。",
            {
                "scene_id": scene_id,
                "source_text": next_scene.get("source_text", ""),
                "summary": next_scene.get("summary", ""),
                "rewrite_reason": "",
            },
        )

    # 只要场景草稿发生过变化且尚未对“最新版本”做 review，就先 review。
    if enable_review and last_reviewed_draft_version < draft_version:
        return (
            TOOL_REVIEW,
            "先审查当前版本的场景草稿，检查是否忠于原文、是否需要回炉修改。",
            {},
        )

    latest_review = review_notes[-1] if review_notes else {}
    latest_issues = collect_review_issues(latest_review)
    valid_pending_rewrites = [
        scene_id
        for scene_id in pending_rewrite_scene_ids
        if scene_has_rewrite_budget(scene_id, state)
    ]
    if latest_issues and valid_pending_rewrites:
        target_scene_id = valid_pending_rewrites[0]
        target_scene = find_scene_plan_item(scene_plan, target_scene_id) or {}
        return (
            TOOL_WRITE,
            f"{target_scene_id} 在审查中被标记为存在问题，先回炉重写这一场，再重新审查。",
            {
                "scene_id": target_scene_id,
                "source_text": target_scene.get("source_text", ""),
                "summary": target_scene.get("summary", ""),
                "rewrite_reason": state.get("pending_rewrite_reasons", {}).get(
                    target_scene_id, ""
                ),
            },
        )

    return (TOOL_FINALIZE, "信息已经足够，进入最终剧本整理。", {})


def build_rule_based_plan(state: NovelScriptState) -> dict:
    """当 LLM planner 不可用或输出非法时的兜底计划器。

    意义不在于“聪明”，而在于：
    - 保证链路永远能跑完
    - 让调试时能分辨是 planner 智能问题，还是底层工具/状态流问题
    """

    tool_name, thought, overrides = _decide_rule_based_action(state)
    tool_input = (
        {}
        if tool_name == TOOL_FINALIZE
        else normalize_tool_input(tool_name, overrides, state)
    )
    return {
        "thought": thought,
        "selected_tool": tool_name,
        "tool_input": tool_input,
    }

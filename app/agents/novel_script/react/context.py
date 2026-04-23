"""planner / 事实抽取所用的上下文构造。

把“把大段原始 state 压缩成少量关键字段”的逻辑单独放一层：
- planner 每轮只看最少必要信息，避免长原文反复进模型
- 事实抽取优先基于场景摘要而不是整本原文

这些函数只读 state，不修改 state，也不打 timing 点——
打点放在 nodes.py 里，这样函数本身保持纯函数、好测试。
"""

from __future__ import annotations

import json

from app.agents.novel_script.constants import (
    DEFAULT_TARGET_SCENE_COUNT,
    FACT_SOURCE_TEXT_MAX_CHARS,
    PLANNER_FACT_CHARACTER_LIMIT,
    PLANNER_FACT_CONFLICT_LIMIT,
    PLANNER_FACT_GOAL_LIMIT,
    PLANNER_FACT_LOCATION_LIMIT,
    PLANNER_PENDING_REWRITE_PREVIEW_LIMIT,
    PLANNER_REVIEW_ISSUE_LIMIT,
    PLANNER_REVIEW_ISSUE_PREVIEW_CHARS,
    PLANNER_REVIEW_TARGET_REASON_PREVIEW_CHARS,
    PLANNER_SCENE_DRAFT_PREVIEW_CHARS,
    PLANNER_SCENE_PREVIEW_LIMIT,
    PLANNER_SCENE_SUMMARY_PREVIEW_CHARS,
    PLANNER_SOURCE_PREVIEW_CHARS,
)
from app.agents.novel_script.state import NovelScriptState

from .script_parse import extract_scene_script_text


def build_fact_source_text(state: NovelScriptState) -> str:
    """构造“事实抽取”阶段真正要送给模型的文本。

    设计原因：
    1. 事实抽取不一定要反复读取完整原文，尤其是长小说片段会显著增加 token 和延迟。
    2. 如果我们已经有 `scene_plan`，说明文本已经被拆成了更稳定的结构，此时优先用场景摘要做抽取，
       往往已经足够覆盖角色、地点、目标这类高层信息。
    3. 如果还没有拆场景，则退回到原文截断版本，保证流程仍可启动。
    """

    scene_plan = state.get("scene_plan", [])
    if scene_plan:
        return "\n".join(
            f"{item.get('scene_id', '')}: {item.get('summary', '')}"
            for item in scene_plan
        )
    return state.get("source_text", "")[:FACT_SOURCE_TEXT_MAX_CHARS]


def build_planner_context(state: NovelScriptState) -> str:
    """构造 ReAct planner 的最小上下文。

    这个方法的目标不是“把所有状态都喂给模型”，而是只保留下一步决策真正需要的关键信息：
    - 原文截断
    - 当前轮次
    - 已拆出的 scene_plan
    - 已生成的 scene_drafts 预览
    - 最近一次 review 结果

    这样做的核心收益：
    1. 降低 planner 每轮重复阅读长文本的成本
    2. 减少无关上下文导致的决策漂移
    3. 让 ReAct 的“想一步、做一步”保持清晰和可调试
    """

    scene_plan = state.get("scene_plan", [])
    scene_drafts = state.get("scene_drafts", [])
    review_notes = state.get("review_notes", [])
    latest_review = review_notes[-1] if review_notes else {}
    story_facts = state.get("story_facts", {}) or {}
    pending_rewrite_scene_ids = state.get("pending_rewrite_scene_ids", []) or []

    # 当还没 split 时，给 planner 一小段原文预览帮助它识别任务类型；
    # 一旦 scene_plan 已存在，就不再重复喂大段原文，改用场景摘要替代。
    source_preview = (state.get("source_text", "") or "").strip().replace("\n", " ")
    source_preview = source_preview[:PLANNER_SOURCE_PREVIEW_CHARS]

    compact_scene_plan = [
        f"{item.get('scene_id', '')}: "
        f"{str(item.get('summary', '')).strip()[:PLANNER_SCENE_SUMMARY_PREVIEW_CHARS]}"
        for item in scene_plan
        if item.get("scene_id")
    ]
    compact_scene_drafts = [
        f"{item.get('scene_id', '')}: "
        f"{extract_scene_script_text(item)[:PLANNER_SCENE_DRAFT_PREVIEW_CHARS]}"
        for item in scene_drafts
        if item.get("scene_id")
    ]
    review_issues = [
        str(item).strip()[:PLANNER_REVIEW_ISSUE_PREVIEW_CHARS]
        for item in (latest_review.get("issues") or [])[:PLANNER_REVIEW_ISSUE_LIMIT]
        if str(item).strip()
    ]
    review_targets = [
        (
            f"{str(item.get('scene_id', '')).strip()}: "
            f"{str(item.get('reason', '')).strip()[:PLANNER_REVIEW_TARGET_REASON_PREVIEW_CHARS]}"
        )
        for item in (latest_review.get("scene_targets") or [])[
            :PLANNER_REVIEW_ISSUE_LIMIT
        ]
        if isinstance(item, dict) and str(item.get("scene_id", "")).strip()
    ]
    fact_summary = {
        "characters": (story_facts.get("characters") or [])[
            :PLANNER_FACT_CHARACTER_LIMIT
        ],
        "locations": (story_facts.get("locations") or [])[
            :PLANNER_FACT_LOCATION_LIMIT
        ],
        "goals": (story_facts.get("goals") or [])[:PLANNER_FACT_GOAL_LIMIT],
        "conflicts": (story_facts.get("conflicts") or [])[
            :PLANNER_FACT_CONFLICT_LIMIT
        ],
    }

    return f"""
任务目标：
{state.get("task_goal", "把小说片段改写成剧本")}

当前状态：
- iteration_count: {state.get("iteration_count", 0)}
- max_iterations: {state.get("max_iterations", 6)}
- target_scene_count: {state.get("target_scene_count", DEFAULT_TARGET_SCENE_COUNT)}
- enable_review: {state.get("enable_review", True)}
- scene_plan_count: {len(scene_plan)}
- scene_draft_count: {len(scene_drafts)}
- pending_rewrite_scene_ids: {json.dumps(pending_rewrite_scene_ids[:PLANNER_PENDING_REWRITE_PREVIEW_LIMIT], ensure_ascii=False)}

故事事实：
{json.dumps(fact_summary, ensure_ascii=False)}

原文预览：
{source_preview if not scene_plan else "已完成场景拆分，planner 无需重复阅读长原文。"}

场景计划摘要：
{json.dumps(compact_scene_plan[:PLANNER_SCENE_PREVIEW_LIMIT], ensure_ascii=False)}

已生成场景摘要：
{json.dumps(compact_scene_drafts[:PLANNER_SCENE_PREVIEW_LIMIT], ensure_ascii=False)}

最近一次审查 issues：
{json.dumps(review_issues, ensure_ascii=False)}

最近一次审查 targets：
{json.dumps(review_targets, ensure_ascii=False)}
""".strip()

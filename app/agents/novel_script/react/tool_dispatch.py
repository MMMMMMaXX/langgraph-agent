"""Tool 调度层。

本模块是 novel_script agent 的“动作执行中枢”，单点管理：
1. **工具名常量**（TOOL_*）+ 合法名单（VALID_SELECTED_TOOLS）
   避免 planner / tool_executor / rule-based fallback 各自维护一份字符串。
2. **工具函数注册表**（TOOLS）：tool_name → 可调用对象
3. **LLM profile 映射**（get_tool_llm_profile）
4. **tool_input 规范化**（normalize_tool_input / filter_tool_input）
5. **状态 reducer**（TOOL_REDUCERS）：每个 tool 的返回值如何 merge 回 state

第 5 项是从 `tool_executor_node` 抽出的：原来节点里用一大段 if/elif
手工合并状态，新增/修改 tool 很容易漏改。改成 dispatch 表后，
加新 tool 只改这一个模块，node 保持最小。
"""

from __future__ import annotations

import inspect
from typing import Callable

from app.agents.novel_script.constants import DEFAULT_TARGET_SCENE_COUNT
from app.agents.novel_script.state import NovelScriptState
from app.agents.novel_script.tools import (
    extract_story_facts,
    review_script,
    split_into_scenes,
    write_script_scene,
)
from app.constants.model_profiles import (
    PROFILE_CREATIVE_PLANNER,
    PROFILE_CREATIVE_REVIEW,
    PROFILE_CREATIVE_WRITE,
    PROFILE_DEFAULT_CHAT,
)

from .context import build_fact_source_text
from .review import build_review_reason_map, extract_review_scene_targets

# ---------- 工具名常量 ----------

TOOL_SPLIT = "split_into_scenes"
TOOL_FACTS = "extract_story_facts"
TOOL_WRITE = "write_script_scene"
TOOL_REVIEW = "review_script"
TOOL_FINALIZE = "finalize"

TOOLS: dict[str, Callable] = {
    TOOL_SPLIT: split_into_scenes,
    TOOL_FACTS: extract_story_facts,
    TOOL_WRITE: write_script_scene,
    TOOL_REVIEW: review_script,
}

# planner 输出的 selected_tool 合法集合——包含 finalize，因为它是有效终态。
VALID_SELECTED_TOOLS: set[str] = set(TOOLS) | {TOOL_FINALIZE}


# ---------- 辅助 ----------


def get_tool_llm_profile(tool_name: str) -> str:
    """把 novel_script 内部动作映射到对应的 LLM profile。

    不是每个 tool 都会调模型：
    - split_into_scenes 目前是规则切分
    - extract_story_facts / write_script_scene / review_script 会调模型

    这里集中映射，便于统一打日志和后续改路由。
    """

    mapping = {
        TOOL_FACTS: PROFILE_DEFAULT_CHAT,
        TOOL_WRITE: PROFILE_CREATIVE_WRITE,
        TOOL_REVIEW: PROFILE_CREATIVE_REVIEW,
        "planner": PROFILE_CREATIVE_PLANNER,
    }
    return mapping.get(tool_name, "")


def find_scene_plan_item(scene_plan: list[dict], scene_id: str) -> dict | None:
    for item in scene_plan:
        if item.get("scene_id") == scene_id:
            return item
    return None


# ---------- tool_input 归一化 ----------


def normalize_tool_input(
    tool_name: str,
    tool_input: dict,
    state: NovelScriptState,
) -> dict:
    """为 planner 产出的工具参数做兜底补全。

    ReAct planner 是“高层决策者”，但不应该被完全信任。
    在长文本场景下，模型有时会输出不完整的 tool_input。
    这里统一做一次参数兜底，把执行层变成更稳定的“守门员”：
    - 自动补齐 text / story_facts / summary / source_text 等必需字段
    - 调用方可以通过传入 overrides 指定具体 scene_id，其它字段再按 state 推断

    setdefault 的选择保证“调用方显式传入 > 自动推断”的优先级不被打破。
    """

    normalized = dict(tool_input)

    if tool_name == TOOL_SPLIT:
        normalized.setdefault("text", state.get("source_text", ""))
        normalized.setdefault(
            "target_scene_count",
            state.get("target_scene_count", DEFAULT_TARGET_SCENE_COUNT),
        )
        return normalized

    if tool_name == TOOL_FACTS:
        normalized.setdefault("text", build_fact_source_text(state))
        return normalized

    if tool_name == TOOL_WRITE:
        scene_drafts = state.get("scene_drafts", [])
        scene_plan = state.get("scene_plan", [])
        rewrite_reasons = state.get("pending_rewrite_reasons", {})
        next_scene = None
        if scene_plan and len(scene_drafts) < len(scene_plan):
            next_scene = scene_plan[len(scene_drafts)]

        normalized.setdefault(
            "scene_id",
            (next_scene or {}).get("scene_id", f"scene_{len(scene_drafts) + 1}"),
        )
        normalized.setdefault(
            "source_text", (next_scene or {}).get("source_text", "")
        )
        normalized.setdefault("summary", (next_scene or {}).get("summary", ""))
        normalized.setdefault("story_facts", state.get("story_facts", {}))
        normalized.setdefault("script_style", state.get("script_style", "影视短剧"))
        normalized.setdefault(
            "rewrite_reason",
            rewrite_reasons.get(normalized.get("scene_id", ""), ""),
        )
        return normalized

    if tool_name == TOOL_REVIEW:
        normalized.setdefault("source_text", state.get("source_text", ""))
        normalized.setdefault("scene_drafts", state.get("scene_drafts", []))
        normalized.setdefault("story_facts", state.get("story_facts", {}))
        return normalized

    return normalized


def filter_tool_input(tool_name: str, tool_input: dict) -> dict:
    """过滤掉不在工具函数签名里的参数。

    planner 偶尔会带上工具签名之外的辅助字段（focus / tone 等）。
    这些字段对“思考”有帮助，但不应该直接传给 Python 工具函数，否则 TypeError。
    这里用反射读取真实函数签名，只保留可执行层真正接受的参数。
    """

    tool_fn = TOOLS[tool_name]
    allowed_keys = set(inspect.signature(tool_fn).parameters)
    return {key: value for key, value in tool_input.items() if key in allowed_keys}


# ---------- tool 结果 -> state 增量 ----------
#
# 每个 reducer 返回一个“部分 state”，由调用方 {**state, **delta} 合并。
# 这样逻辑保持“单个 tool 一个函数”，新增 tool 不用改 node 主流程。


def apply_split_result(result: dict, state: NovelScriptState) -> dict:
    return {"scene_plan": result.get("scenes", [])}


def apply_facts_result(result: dict, state: NovelScriptState) -> dict:
    return {"story_facts": result}


def apply_write_scene_result(result: dict, state: NovelScriptState) -> dict:
    """处理 write_script_scene 的结果。

    写入/覆盖 scene_drafts；若是覆盖则视为“重写”，rewrite_attempts 计数 +1；
    同时把该 scene_id 从 pending_rewrite_scene_ids / reasons 中摘除；
    最后递增 draft_version，让下一轮 review 知道有新版本需要审查。
    """

    scene_drafts = list(state.get("scene_drafts", []))
    scene_id = result.get("scene_id", "")
    existing_index = next(
        (
            index
            for index, item in enumerate(scene_drafts)
            if item.get("scene_id") == scene_id
        ),
        -1,
    )
    rewrite_attempts = dict(state.get("scene_rewrite_attempts", {}))
    pending_rewrites = [
        sid
        for sid in state.get("pending_rewrite_scene_ids", [])
        if sid != scene_id
    ]
    rewrite_reasons = dict(state.get("pending_rewrite_reasons", {}))
    rewrite_reasons.pop(scene_id, None)

    delta: dict = {
        "scene_drafts": scene_drafts,
        "pending_rewrite_scene_ids": pending_rewrites,
        "pending_rewrite_reasons": rewrite_reasons,
        "draft_version": state.get("draft_version", 0) + 1,
    }
    if existing_index >= 0:
        scene_drafts[existing_index] = result
        rewrite_attempts[scene_id] = rewrite_attempts.get(scene_id, 0) + 1
        delta["scene_rewrite_attempts"] = rewrite_attempts
    else:
        scene_drafts.append(result)
    return delta


def apply_review_result(result: dict, state: NovelScriptState) -> dict:
    review_notes = list(state.get("review_notes", []))
    review_notes.append(result)
    scene_drafts = state.get("scene_drafts", [])
    return {
        "review_notes": review_notes,
        "last_reviewed_draft_version": state.get("draft_version", 0),
        "pending_rewrite_scene_ids": extract_review_scene_targets(
            result, scene_drafts
        ),
        "pending_rewrite_reasons": build_review_reason_map(result, scene_drafts),
    }


TOOL_REDUCERS: dict[str, Callable[[dict, NovelScriptState], dict]] = {
    TOOL_SPLIT: apply_split_result,
    TOOL_FACTS: apply_facts_result,
    TOOL_WRITE: apply_write_scene_result,
    TOOL_REVIEW: apply_review_result,
}

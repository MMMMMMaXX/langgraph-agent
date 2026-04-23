import json
import inspect
import re

from langgraph.graph import StateGraph, START, END

from app.agents.novel_script.prompts import REACT_PLANNER_PROMPT
from app.agents.novel_script.state import NovelScriptState
from app.agents.novel_script.tools import (
    split_into_scenes,
    extract_story_facts,
    write_script_scene,
    review_script,
)
from app.agents.novel_script.constants import (
    FACT_SOURCE_TEXT_MAX_CHARS,
    DEFAULT_TARGET_SCENE_COUNT,
    FINAL_REVIEW_NOTE_LIMIT,
    OBSERVATION_PREVIEW_CHARS,
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
    PLANNER_THOUGHT_PREVIEW_CHARS,
    TOOL_OUTPUT_PREVIEW_CHARS,
)
from app.constants.model_profiles import (
    PROFILE_CREATIVE_PLANNER,
    PROFILE_CREATIVE_REVIEW,
    PROFILE_CREATIVE_WRITE,
    PROFILE_DEFAULT_CHAT,
)
from app.llm import chat, get_profile_runtime_info
from app.utils.errors import build_error_info
from app.utils.logger import now_ms


TOOLS = {
    "split_into_scenes": split_into_scenes,
    "extract_story_facts": extract_story_facts,
    "write_script_scene": write_script_scene,
    "review_script": review_script,
}


def add_timing(
    state: NovelScriptState,
    key: str,
    duration_ms: float,
    extra: dict | None = None,
) -> NovelScriptState:
    # creative agent 的性能问题，通常不是“整条链路慢”，而是某一轮某一步特别慢。
    # 这里同时记录：
    # 1. 总体分阶段累计耗时
    # 2. 每一轮的明细耗时
    # 这样日志里能直接回答“卡在第几轮、哪一步”。
    timing_breakdown_ms = dict(state.get("timing_breakdown_ms", {}))
    timing_breakdown_ms[key] = round(timing_breakdown_ms.get(key, 0.0) + duration_ms, 2)

    iteration_timings = list(state.get("iteration_timings", []))
    item = {
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


def build_fact_source_text(state: NovelScriptState) -> str:
    """构造“事实抽取”阶段真正要送给模型的文本。

    设计原因：
    1. 事实抽取不一定要反复读取完整原文，尤其是长小说片段会显著增加 token 和延迟。
    2. 如果我们已经有 `scene_plan`，说明文本已经被拆成了更稳定的结构，此时优先用场景摘要做抽取，
       往往已经足够覆盖角色、地点、目标这类高层信息。
    3. 如果还没有拆场景，则退回到原文截断版本，保证流程仍可启动。
    """
    # 事实抽取不一定要吃完整原文。
    # 当 scene_plan 已经存在时，直接基于场景摘要抽取，能显著减少 token 开销。
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
    # ReAct planner 每一轮只看“当前最关键的中间状态”，
    # 避免把长原文、完整 scene JSON、完整 review JSON 反复塞给模型。
    # planner 的职责只是“决定下一步动作”，不是重新阅读全文做创作。
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


def derive_title(state: NovelScriptState) -> str:
    """从原文首行推断标题。

    当前规则比较保守：
    - 如果首行本身像“第1章 / 第一章”这种章节标题，就直接拿来作为标题
    - 否则退回到通用标题

    这样可以避免标题生成再走一次 LLM，保持 finalizer 的确定性。
    """
    source_text = (state.get("source_text") or "").strip()
    first_line = source_text.splitlines()[0].strip() if source_text else ""
    if first_line.startswith("第") and "章" in first_line:
        return first_line
    return "小说改编剧本"


def build_character_lines(story_facts: dict) -> list[str]:
    characters = story_facts.get("characters") or []
    if not characters:
        return ["- 暂未明确角色信息"]
    return [f"- **{name}**" for name in characters]


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def extract_named_field(field_name: str, scene_script: str) -> str:
    patterns = [
        rf"\*\*{re.escape(field_name)}：\*\*(.+)",
        rf"{re.escape(field_name)}：(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, scene_script)
        if match:
            return match.group(1).strip()
    return ""


def extract_bullet_values(section_title: str, scene_script: str) -> list[str]:
    pattern = rf"\*\*{re.escape(section_title)}：\*\*\n((?:- .+\n?)*)"
    match = re.search(pattern, scene_script)
    if not match:
        return []
    block = match.group(1).strip()
    if not block:
        return []
    values: list[str] = []
    for line in block.splitlines():
        cleaned = line.strip()
        if not cleaned.startswith("- "):
            continue
        value = cleaned[2:].strip()
        if "：" in value:
            value = value.split("：", 1)[0].strip()
        if "（" in value:
            value = value.split("（", 1)[0].strip()
        values.append(value)
    return values


def collect_all_characters(state: NovelScriptState) -> list[str]:
    """汇总所有剧情段落里出现的角色，并去重保序。

    数据来源分两层：
    1. `story_facts.characters`：来自事实抽取，适合作为全局基础名单
    2. 各个 scene draft 的“人物”段：补齐事实抽取漏掉但在具体剧本中出现的人物

    最终用 `dedupe_keep_order` 保证：
    - 不重复
    - 尽量保留首次出现顺序，方便输出更贴近阅读习惯
    """
    candidates: list[str] = []
    candidates.extend((state.get("story_facts", {}) or {}).get("characters") or [])
    for scene_draft in state.get("scene_drafts", []):
        scene_script = extract_scene_script_text(scene_draft)
        candidates.extend(extract_bullet_values("人物", scene_script))
    return dedupe_keep_order(candidates)


def collect_all_locations(state: NovelScriptState) -> list[str]:
    """汇总所有剧情段落里出现的地点，并去重保序。

    这里和角色汇总的思路一致：
    - 先看事实抽取是否已有明确地点
    - 再从每个场景剧本正文里提取“地点”字段

    这样可以支持“场景级地点”和“全局故事地点”同时存在，最后统一合并成地点列表。
    """
    candidates: list[str] = []
    candidates.extend((state.get("story_facts", {}) or {}).get("locations") or [])
    for scene_draft in state.get("scene_drafts", []):
        scene_script = extract_scene_script_text(scene_draft)
        location = extract_named_field("地点", scene_script)
        if location:
            candidates.append(location)
    return dedupe_keep_order(candidates)


def build_location_lines(locations: list[str]) -> list[str]:
    if not locations:
        return ["- 暂未明确地点信息"]
    return [f"- **{item}**" for item in locations]


def build_plot_segment_list_lines(scene_plan: list[dict]) -> list[str]:
    if not scene_plan:
        return ["- 剧情段落信息不足"]
    lines: list[str] = []
    for idx, item in enumerate(scene_plan, start=1):
        summary = item.get("summary", "") or "待补充"
        lines.append(f"- **剧情段落{idx}**：{summary}")
    return lines


def extract_scene_script_text(scene_draft: dict) -> str:
    return (scene_draft.get("script") or "").strip()


def get_tool_llm_profile(tool_name: str) -> str:
    """把 novel_script 内部动作映射到对应的 LLM profile。

    不是每个 tool 都会调模型：
    - split_into_scenes 目前是规则切分
    - extract_story_facts / write_script_scene / review_script 会调模型

    这里集中映射，便于统一打日志和后续改路由。
    """

    mapping = {
        "extract_story_facts": PROFILE_DEFAULT_CHAT,
        "write_script_scene": PROFILE_CREATIVE_WRITE,
        "review_script": PROFILE_CREATIVE_REVIEW,
        "planner": PROFILE_CREATIVE_PLANNER,
    }
    return mapping.get(tool_name, "")


def find_scene_plan_item(scene_plan: list[dict], scene_id: str) -> dict | None:
    for item in scene_plan:
        if item.get("scene_id") == scene_id:
            return item
    return None


def collect_review_issues(review_summary: dict) -> list[str]:
    return [
        str(item).strip()
        for item in (review_summary.get("issues") or [])
        if str(item).strip()
    ]


def extract_review_scene_targets(
    review_summary: dict,
    scene_drafts: list[dict],
) -> list[str]:
    """从 review 结果里提取需要回炉重写的场景列表。

    优先级：
    1. 使用 review_script 显式返回的 `scene_targets`
    2. 从 issues / suggestions 文本里解析 scene_1 / 场景1 / 剧情段落1
    3. 如果 review 确认有问题但无法定位，则保守地让所有场景进入待重写队列
    """
    known_scene_ids = [
        item.get("scene_id", "") for item in scene_drafts if item.get("scene_id")
    ]
    if not known_scene_ids:
        return []

    targets: list[str] = []
    explicit_targets = review_summary.get("scene_targets") or []
    for item in explicit_targets:
        if isinstance(item, dict):
            scene_id = str(item.get("scene_id", "")).strip()
        else:
            scene_id = str(item).strip()
        if scene_id in known_scene_ids and scene_id not in targets:
            targets.append(scene_id)

    if targets:
        return targets

    review_text = "\n".join(
        str(item)
        for item in (
            (review_summary.get("issues") or [])
            + (review_summary.get("suggestions") or [])
        )
    )
    for match in re.finditer(
        r"scene[_\s]?(\d+)|场景\s*(\d+)|剧情段落\s*(\d+)", review_text
    ):
        number_text = next((group for group in match.groups() if group), "")
        if not number_text:
            continue
        scene_id = f"scene_{number_text}"
        if scene_id in known_scene_ids and scene_id not in targets:
            targets.append(scene_id)

    if targets:
        return targets

    if collect_review_issues(review_summary):
        return known_scene_ids

    return []


def build_review_reason_map(
    review_summary: dict,
    scene_drafts: list[dict],
) -> dict[str, str]:
    """把 review 的结果整理成 scene_id -> 返修理由。

    优先使用 review 显式返回的 scene_targets.reason；
    如果 review 没给到足够细的理由，就退回到 issues 的聚合文本，
    至少保证重写时知道“为什么要改”。
    """
    known_scene_ids = {
        item.get("scene_id", "") for item in scene_drafts if item.get("scene_id")
    }
    issues = collect_review_issues(review_summary)
    default_reason = (
        "；".join(issues[:3]) if issues else "审查发现该场景需要进一步修订。"
    )

    reason_map: dict[str, str] = {}
    for item in review_summary.get("scene_targets") or []:
        if not isinstance(item, dict):
            continue
        scene_id = str(item.get("scene_id", "")).strip()
        reason = str(item.get("reason", "")).strip() or default_reason
        if scene_id in known_scene_ids:
            reason_map[scene_id] = reason

    for scene_id in extract_review_scene_targets(review_summary, scene_drafts):
        reason_map.setdefault(scene_id, default_reason)

    return reason_map


def build_adaptation_notes(review_summary: dict, scene_count: int) -> list[str]:
    issues = review_summary.get("issues") or []
    suggestions = review_summary.get("suggestions") or []
    notes: list[str] = [
        f"1. 本次输出采用{'单剧情段落' if scene_count <= 1 else '多剧情段落'}确定性组装，直接复用已生成场景草稿，避免 finalizer 再次整篇重写。",
        "2. 角色表、地点列表和剧情段落列表均基于已生成内容自动汇总并去重。",
    ]
    if issues or suggestions:
        notes.append("3. 已参考审查结果进行整理，审查重点如下：")
        for item in (issues + suggestions)[:FINAL_REVIEW_NOTE_LIMIT]:
            cleaned = str(item).strip()
            if cleaned.startswith("```"):
                continue
            notes.append(f"- {cleaned}")
    return notes


def remove_duplicate_scene_headers(scene_script: str) -> str:
    lines = scene_script.splitlines()
    filtered: list[str] = []
    prefixes = ("**场景ID：", "**场景摘要：", "**剧本正文：**")
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in prefixes):
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()


def build_plot_segments(scene_drafts: list[dict]) -> list[str]:
    """把多个 scene draft 组装成最终正文里的“剧情段落”。

    注意这里故意不再做语义重写，只做结构整理：
    - 去掉重复的场景头
    - 给每段补一个稳定编号
    - 保留原场景正文内容

    这是当前性能优化的关键点之一：
    finalizer 不再把整份剧本丢给 LLM 重写，而是直接复用前面昂贵步骤生成好的场景草稿。
    """
    segments: list[str] = []
    for idx, scene_draft in enumerate(scene_drafts, start=1):
        scene_script = remove_duplicate_scene_headers(
            extract_scene_script_text(scene_draft)
        )
        segments.append(
            "\n".join(
                [
                    f"### 剧情段落{idx}",
                    "",
                    scene_script or "剧情段落内容不足",
                ]
            ).strip()
        )
    return segments


def build_deterministic_final_script(state: NovelScriptState) -> str:
    """确定性地组装最终剧本。

    这一步是当前 novel_script agent 的“轻 finalizer”：
    - 不做 LLM 融合
    - 不做全文改写
    - 只汇总标题、角色、地点、剧情段落列表、正文和改编说明

    这样设计的原因：
    1. 避免最后一跳再次成为整链路最慢的瓶颈
    2. 保证输出结构稳定，便于评测
    3. 让 ReAct 的耗时集中在真正需要创造性生成的步骤
    """
    # 无论单场景还是多场景，finalizer 都只做“组装”而不再做“重写”。
    # 这样能把性能瓶颈锁定在真正的创作步骤：facts / write_scene / review。
    scene_drafts = state.get("scene_drafts", [])
    scene_plan = state.get("scene_plan", [])
    review_notes = state.get("review_notes", [])
    title = derive_title(state)
    review_summary = review_notes[-1] if review_notes else {}
    characters = collect_all_characters(state)
    locations = collect_all_locations(state)
    plot_segments = build_plot_segments(scene_drafts)
    full_script_body = (
        "\n\n---\n\n".join(plot_segments) if plot_segments else "剧情段落内容不足"
    )
    notes = build_adaptation_notes(review_summary, len(plot_segments))

    sections = [
        f"# 《{title}》",
        "",
        "## 改编标题",
        title,
        "",
        "## 角色表",
        *build_character_lines({"characters": characters}),
        "",
        "## 地点列表",
        *build_location_lines(locations),
        "",
        "## 剧情段落列表",
        *build_plot_segment_list_lines(scene_plan),
        "",
        "## 正文剧本",
        full_script_body,
        "",
        "## 改编说明",
        *notes,
    ]
    return "\n".join(sections).strip()


def normalize_tool_input(
    tool_name: str, tool_input: dict, state: NovelScriptState
) -> dict:
    """为 planner 产出的工具参数做兜底补全。

    ReAct planner 负责“决定做什么”，但模型输出的 `tool_input` 可能缺字段、缺上下文，
    所以执行前必须经过这一层守门：
    - 自动补齐 text / story_facts / summary / source_text 等必需字段
    - 把执行层变成稳定接口，而不是完全信任 planner 原始输出
    """
    # ReAct planner 是“高层决策者”，但不应该被完全信任。
    # 在长文本场景下，模型有时会输出不完整的 tool_input。
    # 这里统一做一次参数兜底，把执行层变成更稳定的“守门员”。
    normalized = dict(tool_input)

    if tool_name == "split_into_scenes":
        normalized.setdefault("text", state.get("source_text", ""))
        normalized.setdefault(
            "target_scene_count",
            state.get("target_scene_count", DEFAULT_TARGET_SCENE_COUNT),
        )
        return normalized

    if tool_name == "extract_story_facts":
        normalized.setdefault("text", build_fact_source_text(state))
        return normalized

    if tool_name == "write_script_scene":
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
        normalized.setdefault("source_text", (next_scene or {}).get("source_text", ""))
        normalized.setdefault("summary", (next_scene or {}).get("summary", ""))
        normalized.setdefault("story_facts", state.get("story_facts", {}))
        normalized.setdefault("script_style", state.get("script_style", "影视短剧"))
        normalized.setdefault(
            "rewrite_reason",
            rewrite_reasons.get(normalized.get("scene_id", ""), ""),
        )
        return normalized

    if tool_name == "review_script":
        normalized.setdefault("source_text", state.get("source_text", ""))
        normalized.setdefault("scene_drafts", state.get("scene_drafts", []))
        normalized.setdefault("story_facts", state.get("story_facts", {}))
        return normalized

    return normalized


def filter_tool_input(tool_name: str, tool_input: dict) -> dict:
    """过滤掉不在工具函数签名里的参数。

    典型场景：
    planner 可能为了表达思考，顺手输出 `focus`、`tone` 这类额外字段；
    这些字段对 reasoning 有帮助，但如果直接传给 Python 工具函数就会报错。
    因此这里用反射读取真实函数签名，只保留可执行层真正接受的参数。
    """
    # planner 偶尔会带上工具签名之外的辅助字段，比如 focus / tone。
    # 这些字段对“思考”有帮助，但不应该直接传给 Python 工具函数。
    tool_fn = TOOLS[tool_name]
    allowed_keys = set(inspect.signature(tool_fn).parameters)
    return {key: value for key, value in tool_input.items() if key in allowed_keys}


def build_rule_based_plan(state: NovelScriptState) -> dict:
    """当 LLM planner 不可用或输出非法时的兜底计划器。

    它把整个创作流程固定成一条稳定路径：
    1. 先拆场景
    2. 再抽故事事实
    3. 然后逐场写剧本
    4. 必要时做 review
    5. 最后 finalize

    这个方法的意义不在于“聪明”，而在于：
    - 保证链路永远能跑完
    - 让调试时能分辨是 planner 智能问题，还是底层工具/状态流问题
    """
    # 创作型 ReAct 并不意味着每轮都必须再调用一次 LLM 做“下一步规划”。
    # 当工作流已经非常清晰时，用确定性 planner 更稳、更快，也更便于调试。
    scene_plan = state.get("scene_plan", [])
    story_facts = state.get("story_facts", {})
    scene_drafts = state.get("scene_drafts", [])
    review_notes = state.get("review_notes", [])
    pending_rewrite_scene_ids = list(state.get("pending_rewrite_scene_ids", []))
    scene_rewrite_attempts = dict(state.get("scene_rewrite_attempts", {}))
    max_scene_rewrite_attempts = state.get("max_scene_rewrite_attempts", 1)
    enable_review = state.get("enable_review", True)
    draft_version = state.get("draft_version", 0)
    last_reviewed_draft_version = state.get("last_reviewed_draft_version", -1)

    if not scene_plan:
        return {
            "thought": "先拆分场景，建立改编骨架。",
            "selected_tool": "split_into_scenes",
            "tool_input": {
                "text": state.get("source_text", ""),
                "target_scene_count": state.get(
                    "target_scene_count", DEFAULT_TARGET_SCENE_COUNT
                ),
            },
        }

    if not story_facts:
        return {
            "thought": "先抽取故事事实，避免后续改编偏离原文。",
            "selected_tool": "extract_story_facts",
            "tool_input": {"text": state.get("source_text", "")},
        }

    if len(scene_drafts) < len(scene_plan):
        next_scene = scene_plan[len(scene_drafts)]
        return {
            "thought": f"开始生成 {next_scene.get('scene_id', '下一场')} 的剧本草稿。",
            "selected_tool": "write_script_scene",
            "tool_input": {
                "scene_id": next_scene.get(
                    "scene_id", f"scene_{len(scene_drafts) + 1}"
                ),
                "source_text": next_scene.get("source_text", ""),
                "summary": next_scene.get("summary", ""),
                "story_facts": story_facts,
                "script_style": state.get("script_style", "影视短剧"),
                "rewrite_reason": "",
            },
        }

    # 只要场景草稿发生过变化且尚未对“最新版本”做 review，就先 review。
    if enable_review and last_reviewed_draft_version < draft_version:
        return {
            "thought": "先审查当前版本的场景草稿，检查是否忠于原文、是否需要回炉修改。",
            "selected_tool": "review_script",
            "tool_input": {
                "source_text": state.get("source_text", ""),
                "scene_drafts": scene_drafts,
                "story_facts": story_facts,
            },
        }

    latest_review = review_notes[-1] if review_notes else {}
    latest_issues = collect_review_issues(latest_review)
    valid_pending_rewrites = [
        scene_id
        for scene_id in pending_rewrite_scene_ids
        if scene_rewrite_attempts.get(scene_id, 0) < max_scene_rewrite_attempts
    ]
    if latest_issues and valid_pending_rewrites:
        target_scene_id = valid_pending_rewrites[0]
        target_scene = find_scene_plan_item(scene_plan, target_scene_id) or {}
        return {
            "thought": f"{target_scene_id} 在审查中被标记为存在问题，先回炉重写这一场，再重新审查。",
            "selected_tool": "write_script_scene",
            "tool_input": {
                "scene_id": target_scene_id,
                "source_text": target_scene.get("source_text", ""),
                "summary": target_scene.get("summary", ""),
                "story_facts": story_facts,
                "script_style": state.get("script_style", "影视短剧"),
                "rewrite_reason": state.get("pending_rewrite_reasons", {}).get(
                    target_scene_id, ""
                ),
            },
        }

    return {
        "thought": "信息已经足够，进入最终剧本整理。",
        "selected_tool": "finalize",
        "tool_input": {},
    }


def has_unreviewed_draft(state: NovelScriptState) -> bool:
    """判断当前最新版草稿是否还没经过 review。

    这个判断用于迭代上限兜底：
    如果 rewrite 刚刚改动了场景，`draft_version` 会递增；
    此时即使已经接近 max_iterations，也应优先允许一次复审，
    否则 review 发现的问题会被“改了但没验收”。
    """
    if not state.get("enable_review", True):
        return False
    if not state.get("scene_drafts"):
        return False
    return state.get("last_reviewed_draft_version", -1) < state.get(
        "draft_version", 0
    )


def parse_planner_answer(answer: str) -> dict | None:
    """解析 planner 输出的 JSON，并做最基础的合法性校验。

    校验项只有两类：
    1. JSON 是否可解析
    2. selected_tool 是否属于允许的工具集合

    如果不满足，直接返回 None，让上层回退到 rule-based planner。
    """
    try:
        parsed = json.loads(answer)
    except Exception:
        return None

    selected_tool = parsed.get("selected_tool")
    if selected_tool not in {
        "split_into_scenes",
        "extract_story_facts",
        "write_script_scene",
        "review_script",
        "finalize",
    }:
        return None

    tool_input = parsed.get("tool_input", {})
    if not isinstance(tool_input, dict):
        tool_input = {}

    return {
        "thought": str(parsed.get("thought", "")).strip(),
        "selected_tool": selected_tool,
        "tool_input": tool_input,
    }


def planner_node(state: NovelScriptState) -> NovelScriptState:
    """ReAct graph 中的“思考”节点。

    本节点职责：
    - 调用 LLM planner 决定下一步动作
    - 把 thought / selected_tool / tool_input 写回状态
    - 如果 planner 输出坏掉，则自动切到 rule-based fallback
    - 记录 planner 阶段耗时，便于后续分析哪一步最慢
    """
    started_at_ms = now_ms()
    planner_error = ""
    planner_mode = "react_llm"
    planner_answer = ""
    planner_llm_info = get_profile_runtime_info(PROFILE_CREATIVE_PLANNER)

    # 细拆 planner 阶段，方便定位“到底慢在上下文构造、模型响应，还是解析失败后的兜底”。
    context_started_at_ms = now_ms()
    planner_context = build_planner_context(state)
    context_duration_ms = now_ms() - context_started_at_ms
    timing_state = add_timing(
        state,
        "planner_context_build",
        context_duration_ms,
        extra={
            "context_chars": len(planner_context),
            "scene_count": len(state.get("scene_plan", [])),
            "draft_count": len(state.get("scene_drafts", [])),
            "review_count": len(state.get("review_notes", [])),
            **planner_llm_info,
        },
    )

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
            "planner_llm_call",
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
            "planner_parse",
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
            "planner_llm_call",
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

    if plan is None:
        # ReAct 的好处是有自主决策能力，但工程上仍需要确定性兜底。
        # 当模型超时、限流、输出坏 JSON 或工具名非法时，都回退到规则计划器，确保链路继续执行。
        fallback_started_at_ms = now_ms()
        plan = build_rule_based_plan(timing_state)
        fallback_duration_ms = now_ms() - fallback_started_at_ms
        timing_state = add_timing(
            timing_state,
            "planner_fallback_plan",
            fallback_duration_ms,
            extra={
                "selected_tool": plan.get("selected_tool", "finalize"),
                "reason": planner_error or "invalid planner output",
            },
        )
        if not planner_error:
            planner_mode = "react_fallback_rule"

    next_state: NovelScriptState = {
        **timing_state,
        "thought": plan.get("thought", ""),
        "selected_tool": plan.get("selected_tool", "finalize"),
        "tool_input": plan.get("tool_input", {}),
        "iteration_count": state.get("iteration_count", 0) + 1,
    }
    return add_timing(
        next_state,
        "planner",
        now_ms() - started_at_ms,
        extra={
            "mode": planner_mode,
            "selected_tool": next_state.get("selected_tool", "finalize"),
            "thought": next_state.get("thought", "")[:PLANNER_THOUGHT_PREVIEW_CHARS],
            "context_chars": len(planner_context),
            "planner_answer_chars": len(planner_answer),
            **planner_llm_info,
            "error": planner_error,
        },
    )


def tool_executor_node(state: NovelScriptState) -> NovelScriptState:
    """ReAct graph 中的“行动”节点。

    执行流程：
    1. 读取 planner 刚才选择的工具
    2. 补全参数并过滤非法字段
    3. 调用对应 Python 工具函数
    4. 把工具结果写回状态
    5. 更新 scene_plan / story_facts / scene_drafts / review_notes

    这里本质上是把“LLM 决策”翻译为“确定性函数调用”，是 ReAct 从思考进入执行的桥梁。
    """
    started_at_ms = now_ms()
    tool_name = state.get("selected_tool", "finalize")
    tool_input = normalize_tool_input(
        tool_name,
        dict(state.get("tool_input", {})),
        state,
    )
    history = list(state.get("tool_history", []))

    if tool_name == "finalize":
        next_state = {**state, "done": True}
        return add_timing(
            next_state, "tool_executor_finalize", now_ms() - started_at_ms
        )

    tool_input = filter_tool_input(tool_name, tool_input)
    tool_fn = TOOLS[tool_name]
    tool_profile = get_tool_llm_profile(tool_name)
    tool_llm_info = get_profile_runtime_info(tool_profile) if tool_profile else {}
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

    if tool_name == "split_into_scenes":
        next_state["scene_plan"] = result.get("scenes", [])
    elif tool_name == "extract_story_facts":
        next_state["story_facts"] = result
    elif tool_name == "write_script_scene":
        scene_drafts = list(state.get("scene_drafts", []))
        existing_index = next(
            (
                index
                for index, item in enumerate(scene_drafts)
                if item.get("scene_id") == result.get("scene_id")
            ),
            -1,
        )
        rewrite_attempts = dict(state.get("scene_rewrite_attempts", {}))
        pending_rewrites = [
            scene_id
            for scene_id in state.get("pending_rewrite_scene_ids", [])
            if scene_id != result.get("scene_id")
        ]
        rewrite_reasons = dict(state.get("pending_rewrite_reasons", {}))
        rewrite_reasons.pop(result.get("scene_id", ""), None)
        if existing_index >= 0:
            scene_drafts[existing_index] = result
            scene_id = result.get("scene_id", "")
            rewrite_attempts[scene_id] = rewrite_attempts.get(scene_id, 0) + 1
            next_state["scene_rewrite_attempts"] = rewrite_attempts
        else:
            scene_drafts.append(result)
        next_state["scene_drafts"] = scene_drafts
        next_state["pending_rewrite_scene_ids"] = pending_rewrites
        next_state["pending_rewrite_reasons"] = rewrite_reasons
        next_state["draft_version"] = state.get("draft_version", 0) + 1
    elif tool_name == "review_script":
        review_notes = list(state.get("review_notes", []))
        review_notes.append(result)
        next_state["review_notes"] = review_notes
        next_state["last_reviewed_draft_version"] = state.get("draft_version", 0)
        next_state["pending_rewrite_scene_ids"] = extract_review_scene_targets(
            result,
            state.get("scene_drafts", []),
        )
        next_state["pending_rewrite_reasons"] = build_review_reason_map(
            result,
            state.get("scene_drafts", []),
        )

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
    if state.get("done"):
        return "finalize"

    if state.get("selected_tool") == "finalize":
        return "finalize"

    return "planner"


def finalizer_node(state: NovelScriptState) -> NovelScriptState:
    """ReAct graph 的收尾节点。

    当前 finalizer 是纯确定性组装器，不再调用 LLM。
    它只负责：
    - 从 state 中读取 scene_drafts / facts / review 等中间结果
    - 生成最终输出字符串
    - 标记 done=True

    这么做的主要目标是把收尾阶段的耗时压到极低，同时避免“最后一跳重写”破坏前面已经写好的场景。
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
        "finalizer",
        now_ms() - started_at_ms,
        extra={
            "strategy": finalizer_strategy,
            "answer_chars": len(final_script),
        },
    )


builder = StateGraph(NovelScriptState)
builder.add_node("planner", planner_node)
builder.add_node("tool_executor", tool_executor_node)
builder.add_node("finalizer", finalizer_node)

builder.add_edge(START, "planner")
builder.add_conditional_edges(
    "planner",
    should_continue,
    {
        "planner": "tool_executor",
        "finalize": "finalizer",
    },
)
builder.add_conditional_edges(
    "tool_executor",
    should_continue,
    {
        "planner": "planner",
        "finalize": "finalizer",
    },
)
builder.add_edge("finalizer", END)

novel_script_graph = builder.compile()

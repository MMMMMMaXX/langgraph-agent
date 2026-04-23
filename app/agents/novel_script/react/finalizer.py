"""确定性的最终剧本组装。

novel_script 在“创作 → 审查 → 回炉”循环结束后，由 finalizer 把中间产物
组装成一份完整剧本。当前设计坚持“finalizer 不调 LLM”：
- 避免最后一跳再次成为整链路最慢的瓶颈
- 保证输出结构稳定，便于评测
- 让 ReAct 的耗时集中在真正需要创造性生成的步骤

本模块只做数据聚合 + 字符串组装，所有生成步骤都发生在上游（write_script_scene）。
"""

from __future__ import annotations

from app.agents.novel_script.constants import FINAL_REVIEW_NOTE_LIMIT
from app.agents.novel_script.state import NovelScriptState

from .script_parse import (
    dedupe_keep_order,
    extract_bullet_values,
    extract_named_field,
    extract_scene_script_text,
    remove_duplicate_scene_headers,
)


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


def build_location_lines(locations: list[str]) -> list[str]:
    if not locations:
        return ["- 暂未明确地点信息"]
    return [f"- **{item}**" for item in locations]


def collect_all_characters(state: NovelScriptState) -> list[str]:
    """汇总所有剧情段落里出现的角色，并去重保序。

    数据来源分两层：
    1. `story_facts.characters`：来自事实抽取，适合作为全局基础名单
    2. 各个 scene draft 的“人物”段：补齐事实抽取漏掉但在具体剧本中出现的人物

    最终用 `dedupe_keep_order` 保证不重复，并尽量保留首次出现顺序。
    """

    candidates: list[str] = []
    candidates.extend((state.get("story_facts", {}) or {}).get("characters") or [])
    for scene_draft in state.get("scene_drafts", []):
        scene_script = extract_scene_script_text(scene_draft)
        candidates.extend(extract_bullet_values("人物", scene_script))
    return dedupe_keep_order(candidates)


def collect_all_locations(state: NovelScriptState) -> list[str]:
    """汇总所有剧情段落里出现的地点，并去重保序。

    先看事实抽取是否已有明确地点；
    再从每个场景剧本正文里提取“地点”字段。
    这样“场景级地点”和“全局故事地点”可以同时存在，最后统一合并。
    """

    candidates: list[str] = []
    candidates.extend((state.get("story_facts", {}) or {}).get("locations") or [])
    for scene_draft in state.get("scene_drafts", []):
        scene_script = extract_scene_script_text(scene_draft)
        location = extract_named_field("地点", scene_script)
        if location:
            candidates.append(location)
    return dedupe_keep_order(candidates)


def build_plot_segment_list_lines(scene_plan: list[dict]) -> list[str]:
    if not scene_plan:
        return ["- 剧情段落信息不足"]
    lines: list[str] = []
    for idx, item in enumerate(scene_plan, start=1):
        summary = item.get("summary", "") or "待补充"
        lines.append(f"- **剧情段落{idx}**：{summary}")
    return lines


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


def build_deterministic_final_script(state: NovelScriptState) -> str:
    """确定性地组装最终剧本。

    只做汇总：标题、角色、地点、剧情段落列表、正文、改编说明；
    不调用 LLM、不做整篇重写。
    """

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

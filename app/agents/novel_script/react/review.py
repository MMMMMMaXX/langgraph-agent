"""review 结果解读工具。

`review_script` 工具会返回包含 issues / suggestions / scene_targets 的结构化反馈，
这里负责把它翻译成主流程真正要消费的三件事：
1. 哪些 scene_id 需要回炉重写（pending_rewrite_scene_ids）
2. 每个被标记 scene_id 的重写理由（pending_rewrite_reasons）
3. 从 review 自由文本里兜底解析 scene_id（即使模型没给 scene_targets 也不至于拿不到目标）

`scene_has_rewrite_budget` 是 planner 和 tool_executor 共同关心的小判定：
一个场景是不是还能继续返修（未超过 max_scene_rewrite_attempts）。
"""

from __future__ import annotations

import re

from app.agents.novel_script.state import NovelScriptState


def collect_review_issues(review_summary: dict) -> list[str]:
    """把 review 里的 issues 文本去空白、转字符串列表。"""

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


def scene_has_rewrite_budget(scene_id: str, state: NovelScriptState) -> bool:
    """判断某个场景是否还能继续返修。

    Planner 决定下一步时不能无限让同一个场景 review → rewrite → review 循环，
    tool_executor 也在实际执行前做一次判定。集中放在这里避免两处逻辑漂移。
    """

    attempts = (state.get("scene_rewrite_attempts") or {}).get(scene_id, 0)
    max_attempts = state.get("max_scene_rewrite_attempts", 1)
    return attempts < max_attempts

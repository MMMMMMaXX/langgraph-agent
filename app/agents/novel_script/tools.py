import json
import re

from app.llm import chat
from app.prompts.story import (
    EXTRACT_STORY_FACTS_SYSTEM_PROMPT,
    REVIEW_SCRIPT_SYSTEM_PROMPT,
)
from app.prompts.creative import (
    build_write_script_scene_system_prompt,
    build_write_script_scene_user_prompt,
)
from app.agents.novel_script.constants import (
    REVIEW_FACT_CHARACTER_LIMIT,
    REVIEW_FACT_CONFLICT_LIMIT,
    REVIEW_FACT_GOAL_LIMIT,
    REVIEW_FACT_LOCATION_LIMIT,
    REVIEW_MAX_COMPLETION_TOKENS,
    REVIEW_SCENE_DRAFT_PREVIEW_CHARS,
    REVIEW_SOURCE_PREVIEW_CHARS,
    SCENE_SUMMARY_MAX_CHARS,
    WRITE_SCENE_MAX_COMPLETION_TOKENS,
)
from app.constants.model_profiles import PROFILE_CREATIVE_REVIEW, PROFILE_CREATIVE_WRITE


CHAPTER_TITLE_RE = re.compile(
    r"^\s*第(?:\d+|[一二三四五六七八九十百千两零〇]+)[章节回篇卷部集幕]\b.*$"
)


def split_by_chapter_titles(text: str) -> list[str]:
    """按“第1章 / 第一章”这类章节标题切分长文本。

    这是比“按段落均分”更贴近小说结构的高层切分方式。
    一旦命中章节标题，我们优先相信作者给出的天然分段，而不是再人工平均拆块。
    """
    # 对小说文本来说，章节标题本身就是最自然的高层结构。
    # 只要命中“第1章 / 第一章”这种格式，就优先按章节切分，
    # 不再退回到按段落平均分桶的粗糙策略。
    lines = text.splitlines()
    chunks: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        is_chapter_title = bool(CHAPTER_TITLE_RE.match(stripped))

        if is_chapter_title and current_lines:
            chunk = "\n".join(current_lines).strip()
            if chunk:
                chunks.append(chunk)
            current_lines = [line]
            continue

        current_lines.append(line)

    if current_lines:
        chunk = "\n".join(current_lines).strip()
        if chunk:
            chunks.append(chunk)

    if len(chunks) <= 1:
        return []
    return chunks


def split_into_scenes(text: str, target_scene_count: int = 3) -> dict:
    """生成后续 ReAct 流程要使用的 scene_plan。

    处理顺序：
    1. 先尝试按章节标题切分
    2. 如果没有章节标题，再退化成按段落分桶

    输出统一为：
    {
      "scenes": [
        {"scene_id": "...", "summary": "...", "source_text": "..."}
      ]
    }

    这样后续 planner / write_script_scene 都只依赖统一结构，不关心切分策略本身。
    """
    # 第一版先用非常轻量的规则切分，
    # 目标不是“最聪明”，而是先给 ReAct 一个稳定可用的 scene_plan。
    chapter_chunks = split_by_chapter_titles(text)
    if chapter_chunks:
        return {
            "scenes": [
                {
                    "scene_id": f"scene_{idx}",
                    "summary": chunk[:SCENE_SUMMARY_MAX_CHARS],
                    "source_text": chunk,
                }
                for idx, chunk in enumerate(chapter_chunks, start=1)
            ]
        }

    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    if not paragraphs:
        return {"scenes": []}

    target_scene_count = max(1, min(target_scene_count, len(paragraphs)))
    chunk_size = max(1, len(paragraphs) // target_scene_count)

    scenes = []
    current_parts: list[str] = []
    for paragraph in paragraphs:
        current_parts.append(paragraph)
        if len(current_parts) >= chunk_size and len(scenes) < target_scene_count - 1:
            content = "\n".join(current_parts)
            scenes.append(
                {
                    "scene_id": f"scene_{len(scenes) + 1}",
                    "summary": content[:SCENE_SUMMARY_MAX_CHARS],
                    "source_text": content,
                }
            )
            current_parts = []

    if current_parts:
        content = "\n".join(current_parts)
        scenes.append(
            {
                "scene_id": f"scene_{len(scenes) + 1}",
                "summary": content[:SCENE_SUMMARY_MAX_CHARS],
                "source_text": content,
            }
        )

    return {"scenes": scenes}


def extract_story_facts(text: str) -> dict:
    """从原文或场景摘要中抽取后续写作要依赖的故事事实。

    这是 creative agent 里最重要的“约束层”之一：
    - 人物、地点、冲突、目标会成为后续写场景的事实边界
    - 即使场景写作是创造性的，也尽量围绕这些抽取结果展开

    如果模型返回非法 JSON，则保留原始文本到 `raw` 字段，方便调试。
    """
    # 事实抽取是这类创作 agent 的关键步骤：
    # 后续写剧本时要尽量依赖这些事实，减少人设漂移和剧情幻觉。
    answer = chat(
        [
            {
                "role": "system",
                "content": EXTRACT_STORY_FACTS_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        max_completion_tokens=220,
    )

    try:
        return json.loads(answer)
    except Exception:
        return {
            "characters": [],
            "locations": [],
            "conflicts": [],
            "goals": [],
            "raw": answer,
        }


def write_script_scene(
    scene_id: str,
    source_text: str,
    summary: str,
    story_facts: dict,
    script_style: str = "影视短剧",
    rewrite_reason: str = "",
) -> dict:
    """把单个场景的原文片段改写成剧本草稿。

    这是当前链路里最核心、也最耗时的创作步骤之一。
    它只负责“单场景写作”，不负责全文排版，原因是：
    - 单场景更容易 review
    - 单场景更容易局部重写
    - 单场景更容易定位具体是哪一段写得慢、写得偏
    """
    # 这个工具只负责“单场景改写”，不负责整部剧本的排版。
    # 这样 scene 级别的生成、重写、review 都更容易做。
    answer = chat(
        [
            {
                "role": "system",
                "content": build_write_script_scene_system_prompt(script_style),
            },
            {
                "role": "user",
                "content": build_write_script_scene_user_prompt(
                    scene_id=scene_id,
                    summary=summary,
                    story_facts_json=json.dumps(story_facts, ensure_ascii=False),
                    rewrite_reason=rewrite_reason,
                    source_text=source_text,
                ),
            },
        ],
        max_completion_tokens=WRITE_SCENE_MAX_COMPLETION_TOKENS,
        profile=PROFILE_CREATIVE_WRITE,
    )
    return {
        "scene_id": scene_id,
        "script": answer,
        "rewrite_reason": rewrite_reason,
    }


def parse_json_object(answer: str) -> dict | None:
    """从模型回答中解析 JSON 对象。

    creative_review 理论上只输出 JSON，但真实模型偶尔会包一层 ```json。
    这里做轻量清洗，避免因为格式外壳导致 review 结果丢失。
    """
    cleaned = (answer or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_review_result(answer: str, scene_drafts: list[dict]) -> dict:
    """把 review 模型输出归一成稳定结构。

    下游是否触发返修只看 `issues + scene_targets`。
    如果模型没有严格按 JSON 输出，我们也保留 raw 文本，方便 debug；
    但不会凭空制造 issue，避免 review 解析失败时误触发全量重写。
    """
    parsed = parse_json_object(answer)
    if parsed is None:
        return {
            "issues": [],
            "suggestions": [answer],
            "scene_targets": [],
            "raw": answer,
        }

    known_scene_ids = {
        str(item.get("scene_id", "")).strip()
        for item in scene_drafts
        if item.get("scene_id")
    }
    issues = [
        str(item).strip()
        for item in (parsed.get("issues") or [])
        if str(item).strip()
    ]
    suggestions = [
        str(item).strip()
        for item in (parsed.get("suggestions") or [])
        if str(item).strip()
    ]

    scene_targets: list[dict] = []
    for item in parsed.get("scene_targets") or []:
        if isinstance(item, dict):
            scene_id = str(item.get("scene_id", "")).strip()
            reason = str(item.get("reason", "")).strip()
        else:
            scene_id = str(item).strip()
            reason = ""
        if scene_id in known_scene_ids:
            scene_targets.append(
                {
                    "scene_id": scene_id,
                    "reason": reason or "审查发现该场景需要进一步修订。",
                }
            )

    return {
        "issues": issues,
        "suggestions": suggestions,
        "scene_targets": scene_targets,
    }


def review_script(
    source_text: str, scene_drafts: list[dict], story_facts: dict
) -> dict:
    """对已生成的场景草稿做轻量自审。

    review 的定位不是重写，而是发现问题：
    - 事实偏离
    - 人设偏移
    - 场景不连贯
    - 对白不自然

    它的输出会作为后续整理阶段的参考，帮助我们在不增加大规模重写成本的情况下修正明显问题。
    """
    # review_script 相当于一个轻量自检器，
    # 帮助 agent 在 finalize 之前先发现明显的偏离和不自然问题。
    #
    # 这里刻意不再把“完整原文 + 完整场景草稿 JSON”整包塞给模型，
    # 因为 review 的目标只是判断：
    # 1. 是否偏离关键事实
    # 2. 哪一场最值得回炉
    # 所以只保留紧凑摘要即可，能显著降低审查阶段的 token 和耗时。
    source_preview = (source_text or "").strip().replace("\n", " ")[
        :REVIEW_SOURCE_PREVIEW_CHARS
    ]
    compact_facts = {
        "characters": (story_facts.get("characters") or [])[
            :REVIEW_FACT_CHARACTER_LIMIT
        ],
        "locations": (story_facts.get("locations") or [])[
            :REVIEW_FACT_LOCATION_LIMIT
        ],
        "goals": (story_facts.get("goals") or [])[:REVIEW_FACT_GOAL_LIMIT],
        "conflicts": (story_facts.get("conflicts") or [])[
            :REVIEW_FACT_CONFLICT_LIMIT
        ],
    }
    compact_scene_drafts = [
        {
            "scene_id": item.get("scene_id", ""),
            "script_preview": str(item.get("script", "")).strip()[
                :REVIEW_SCENE_DRAFT_PREVIEW_CHARS
            ],
        }
        for item in scene_drafts
        if item.get("scene_id")
    ]

    answer = chat(
        [
            {
                "role": "system",
                "content": REVIEW_SCRIPT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
原文摘要：
{source_preview}

关键事实摘要：
{json.dumps(compact_facts, ensure_ascii=False)}

场景草稿预览：
{json.dumps(compact_scene_drafts, ensure_ascii=False)}
""".strip(),
            },
        ],
        max_completion_tokens=REVIEW_MAX_COMPLETION_TOKENS,
        profile=PROFILE_CREATIVE_REVIEW,
    )

    return normalize_review_result(answer, compact_scene_drafts)

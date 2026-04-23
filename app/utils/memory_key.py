import hashlib

from app.constants.keywords import (
    CORRECTION_MEMORY_KEYWORDS,
    META_MEMORY_KEYWORDS,
    PREFERENCE_MEMORY_KEYWORDS,
    SUMMARY_MEMORY_KEYWORDS,
    TASK_STATE_MEMORY_KEYWORDS,
    contains_any,
)
from app.constants.tags import CITY_TAGS, TOPIC_TAGS

MEMORY_TYPE_FACT = "fact"
MEMORY_TYPE_PREFERENCE = "preference"
MEMORY_TYPE_CORRECTION = "correction"
MEMORY_TYPE_TASK_STATE = "task_state"
MEMORY_TYPE_META = "meta"
MEMORY_TYPE_SUMMARY = "summary"


def _short_text_hash(text: str) -> str:
    normalized = " ".join((text or "").split())
    if not normalized:
        return "empty"
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def build_memory_key(rewritten_query: str, tags: list[str]) -> str:
    city_tags = [t for t in tags if t in CITY_TAGS]
    type_tags = [t for t in tags if t in TOPIC_TAGS]

    if city_tags and type_tags:
        return f"{city_tags[0]}:{type_tags[0]}"

    # 非天气/气候类 memory 可能没有城市或主题标签。
    # 这时不能继续使用 unknown_city:unknown_type，否则多条偏好/纠错/任务状态会互相覆盖。
    city_part = city_tags[0] if city_tags else "general"
    type_part = type_tags[0] if type_tags else "semantic"
    return f"{city_part}:{type_part}:{_short_text_hash(rewritten_query)}"


def dedupe_memory_hits(memory_hits: list[dict]) -> list[dict]:
    seen = set()
    result = []

    for hit in memory_hits:
        key = hit.get("memory_key") or hit.get("content")
        if key in seen:
            continue
        seen.add(key)
        result.append(hit)

    return result


def classify_memory_type(query: str) -> str:
    """把一轮对话粗分成长期记忆类型。

    这里先用轻量关键词规则，保持可解释、可调参。
    后续如果要更工业化，可以把这个函数替换成小模型分类器，
    但外部仍然只依赖这些稳定的 memory_type 常量。
    """

    query = (query or "").strip()

    if contains_any(query, SUMMARY_MEMORY_KEYWORDS):
        return MEMORY_TYPE_SUMMARY

    if contains_any(query, META_MEMORY_KEYWORDS):
        return MEMORY_TYPE_META

    if contains_any(query, CORRECTION_MEMORY_KEYWORDS):
        return MEMORY_TYPE_CORRECTION

    if contains_any(query, PREFERENCE_MEMORY_KEYWORDS):
        return MEMORY_TYPE_PREFERENCE

    if contains_any(query, TASK_STATE_MEMORY_KEYWORDS):
        return MEMORY_TYPE_TASK_STATE

    return MEMORY_TYPE_FACT

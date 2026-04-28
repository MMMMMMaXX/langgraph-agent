"""RAG query 分类。

第一版先用确定性规则，避免为了分类额外增加一次 LLM 调用。分类结果会驱动回答
策略和 debug/eval 指标；后续如果规则不够，再把这里替换成 LLM classifier。
"""

from __future__ import annotations

from app.agents.rag.constants import (
    QUERY_TYPE_COMPARISON,
    QUERY_TYPE_DEFINITION,
    QUERY_TYPE_FACTUAL,
    QUERY_TYPE_FALLBACK,
    QUERY_TYPE_FOLLOWUP,
)
from app.agents.rag.types import QueryClassification
from app.constants.keywords import (
    FOLLOWUP_QUERY_MAX_CHARS,
    FOLLOWUP_QUERY_PREFIXES,
    FOLLOWUP_QUERY_SUFFIXES,
)

DEFINITION_QUERY_KEYWORDS = ("是什么", "什么是", "定义", "概念", "含义")
COMPARISON_QUERY_KEYWORDS = (
    "区别",
    "差异",
    "对比",
    "比较",
    "相比",
    "相同点",
    "不同点",
    "优缺点",
    " vs ",
    " VS ",
)
FOLLOWUP_REFERENCE_KEYWORDS = (
    "它",
    "这个",
    "那个",
    "前者",
    "后者",
    "上面",
    "刚才",
)
VAGUE_QUERY_KEYWORDS = ("这个是什么", "那个是什么", "介绍一下", "讲讲", "说说")


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _looks_like_short_followup(message: str) -> bool:
    stripped = message.strip()
    if len(stripped) > FOLLOWUP_QUERY_MAX_CHARS:
        return False
    return stripped.startswith(FOLLOWUP_QUERY_PREFIXES) or stripped.endswith(
        FOLLOWUP_QUERY_SUFFIXES
    )


def _looks_like_comparison(message: str) -> bool:
    if _contains_any(message, COMPARISON_QUERY_KEYWORDS):
        return True
    return ("和" in message or "与" in message) and (
        "关系" in message or "不同" in message or "一样" in message
    )


def _looks_like_vague_query(message: str, has_context: bool) -> bool:
    stripped = message.strip()
    if not stripped:
        return True
    if len(stripped) <= 2:
        return True
    if _contains_any(stripped, VAGUE_QUERY_KEYWORDS) and not has_context:
        return True
    return False


def classify_rag_query(
    *,
    original_query: str,
    rewritten_query: str,
    has_context: bool,
) -> QueryClassification:
    """按原始 query 和改写 query 生成分类。

    追问要优先看原始 query，因为“那上海呢”改写后会变成完整问题，若只看
    rewritten_query 就丢掉了“这是追问”的信号。
    """

    original = original_query.strip()
    rewritten = rewritten_query.strip()
    combined = f"{original}\n{rewritten}"

    if _looks_like_short_followup(original) or (
        has_context and _contains_any(original, FOLLOWUP_REFERENCE_KEYWORDS)
    ):
        return QueryClassification(
            query_type=QUERY_TYPE_FOLLOWUP,
            confidence=0.9,
            reason="original_query_contains_followup_signal",
        )

    if _looks_like_comparison(combined):
        return QueryClassification(
            query_type=QUERY_TYPE_COMPARISON,
            confidence=0.85,
            reason="query_contains_comparison_signal",
        )

    if _contains_any(combined, DEFINITION_QUERY_KEYWORDS):
        return QueryClassification(
            query_type=QUERY_TYPE_DEFINITION,
            confidence=0.9,
            reason="query_contains_definition_signal",
        )

    if _looks_like_vague_query(original, has_context):
        return QueryClassification(
            query_type=QUERY_TYPE_FALLBACK,
            confidence=0.75,
            reason="query_is_too_short_or_vague",
        )

    return QueryClassification(
        query_type=QUERY_TYPE_FACTUAL,
        confidence=0.6,
        reason="default_factual_query",
    )

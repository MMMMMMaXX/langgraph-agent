"""RAG query 分类。

第一版先用确定性规则，避免为了分类额外增加一次 LLM 调用。分类结果会驱动回答
策略和 debug/eval 指标；后续如果规则不够，再把这里替换成 LLM classifier。
"""

from __future__ import annotations

import json

from app.agents.rag.constants import (
    CLASSIFIER_LLM_CONFIDENCE_THRESHOLD,
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

DEFINITION_QUERY_KEYWORDS = (
    # 概念类
    "是什么",
    "什么是",
    "定义",
    "概念",
    "含义",
    "是什么意思",
    "指的是",
    "什么叫",
    "怎么理解",
    # 操作/使用类：这类问题同样需要"概念 + 关键作用"的定义型回答策略
    "怎么用",
    "如何使用",
    "怎么使用",
    "如何配置",
    "怎么配置",
    "如何设置",
    "怎么设置",
    # 功能/作用类
    "有什么作用",
    "用来做什么",
    "有什么功能",
    "有什么用",
    "能做什么",
)
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
    " Vs ",
)
FOLLOWUP_REFERENCE_KEYWORDS = (
    "它",
    "这",
    "这个",
    "该",
    "那个",
    "前者",
    "后者",
    "上述",
    "上面",
    "刚才",
)
VAGUE_QUERY_KEYWORDS = ("这个是什么", "那个是什么", "介绍一下", "讲讲", "说说")

_VALID_QUERY_TYPES = {
    QUERY_TYPE_DEFINITION,
    QUERY_TYPE_COMPARISON,
    QUERY_TYPE_FACTUAL,
    QUERY_TYPE_FOLLOWUP,
    QUERY_TYPE_FALLBACK,
}


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


# ---------------------------------------------------------------------------
# LLM 二裁（Rules + LLM 混合模式）
# ---------------------------------------------------------------------------


def _should_llm_classify(classification: QueryClassification) -> bool:
    """判断规则分类是否置信度不足，需要 LLM 二裁。"""
    return classification.confidence < CLASSIFIER_LLM_CONFIDENCE_THRESHOLD


def _parse_llm_classification(raw: str) -> QueryClassification | None:
    """解析 LLM 分类响应 JSON，解析失败或类型非法时返回 None。"""
    try:
        data = json.loads(raw.strip())
        query_type = str(data.get("type", "")).lower()
        confidence = float(data.get("confidence", 0.0))
        if query_type not in _VALID_QUERY_TYPES:
            return None
        return QueryClassification(
            query_type=query_type,
            confidence=min(max(confidence, 0.0), 1.0),
            reason="llm_classifier",
        )
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None


def classify_with_llm(query: str) -> QueryClassification | None:
    """用轻量 LLM 对 query 做语义分类。

    仅在规则分类置信度不足时调用（通常为 FACTUAL/FALLBACK）。
    失败时返回 None，调用方回退到规则结果，不中断主链路。

    延迟影响：约 200~500ms（取决于模型），仅发生在低置信度 case。
    """
    # 懒导入避免在纯规则路径上引入 LLM 依赖
    from app.llm import LLMCallError, chat
    from app.llm.providers import PROFILE_CLASSIFY
    from app.prompts.rag import CLASSIFIER_SYSTEM_PROMPT, build_classifier_user_prompt
    from app.utils.logger import log_warning

    try:
        raw = chat(
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": build_classifier_user_prompt(query)},
            ],
            max_completion_tokens=40,
            profile=PROFILE_CLASSIFY,
            trace_stage="query_classify",
        )
        return _parse_llm_classification(raw)
    except LLMCallError as exc:
        log_warning(
            "classify_with_llm",
            "LLM classifier failed; fallback to rule result",
            {
                "code": exc.code,
                "profile": exc.profile,
                "provider": exc.provider,
                "model": exc.model,
                "query_preview": query[:80],
            },
        )
        return None
    except Exception as exc:
        log_warning(
            "classify_with_llm",
            "unexpected error in LLM classifier; fallback to rule result",
            {"error": f"{exc.__class__.__name__}: {exc}", "query_preview": query[:80]},
        )
        return None


def classify_rag_query(
    *,
    original_query: str,
    rewritten_query: str,
    has_context: bool,
    llm_fallback: bool = False,
) -> QueryClassification:
    """按原始 query 和改写 query 生成分类。

    追问要优先看原始 query，因为"那上海呢"改写后会变成完整问题，若只看
    rewritten_query 就丢掉了"这是追问"的信号。

    llm_fallback=True 时：规则置信度 < CLASSIFIER_LLM_CONFIDENCE_THRESHOLD 会触发
    LLM 二裁。建议只在改写后的最终分类阶段开启，预分类（用于决定是否 LLM 改写）
    不需要精确分类，不必付 LLM 成本。
    """

    original = original_query.strip()
    rewritten = rewritten_query.strip()
    combined = f"{original}\n{rewritten}"

    if _looks_like_short_followup(original) or (
        has_context
        and (
            _contains_any(original, FOLLOWUP_REFERENCE_KEYWORDS)
            or _contains_any(original, VAGUE_QUERY_KEYWORDS)
        )
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

    result = QueryClassification(
        query_type=QUERY_TYPE_FACTUAL,
        confidence=0.6,
        reason="default_factual_query",
    )

    # LLM 二裁：仅在置信度不足 + 允许 LLM 时触发
    if llm_fallback and _should_llm_classify(result):
        llm_result = classify_with_llm(rewritten or original)
        if llm_result is not None:
            return llm_result

    return result

"""rag_agent 查询改写辅助函数。"""

from app.agents.rag.constants import QUERY_TYPE_FOLLOWUP
from app.agents.rag.query_classifier import classify_rag_query
from app.agents.rag.types import QueryClassification, RewriteResult
from app.constants.model_profiles import PROFILE_REWRITE
from app.llm import rewrite_query
from app.utils.errors import build_error_info
from app.utils.logger import now_ms


REWRITE_MODE_LLM = "llm"
REWRITE_MODE_SKIP = "skip"
REWRITE_TRIGGER_FOLLOWUP = "followup_query"
REWRITE_SKIP_NO_CONTEXT = "no_context"
REWRITE_SKIP_NOT_FOLLOWUP = "not_followup"


def normalize_query_mark(message: str) -> str:
    """确保检索 query 至少是疑问句形态，不做领域补全。"""

    stripped = message.strip()
    return stripped if stripped.endswith(("？", "?")) else f"{stripped}？"


def get_user_messages(messages: list[dict], limit: int = 2) -> list[dict]:
    """取最近几条用户消息，供 LLM rewrite 补充上下文。"""

    user_msgs = [m for m in messages if m["role"] == "user"]
    return user_msgs[-limit:]


def build_rewrite_messages(
    *,
    messages: list[dict],
    summary: str,
) -> list[dict]:
    """构造 LLM rewrite 输入，包含摘要和最近用户问题。"""

    rewrite_input = []
    if summary:
        rewrite_input.append({"role": "system", "content": f"对话背景：{summary}"})
    rewrite_input.extend(get_user_messages(messages, 3))
    return rewrite_input


def should_llm_rewrite(
    *,
    classification: QueryClassification,
    has_context: bool,
) -> tuple[bool, str, str]:
    """根据 RAG query 分类决定是否需要 LLM 改写。"""

    if not has_context:
        return False, "", REWRITE_SKIP_NO_CONTEXT
    if classification.query_type == QUERY_TYPE_FOLLOWUP:
        return True, REWRITE_TRIGGER_FOLLOWUP, ""
    return False, "", REWRITE_SKIP_NOT_FOLLOWUP


def rewrite_rag_query(
    message: str,
    *,
    messages: list[dict],
    summary: str,
    classification: QueryClassification | None = None,
) -> RewriteResult:
    """改写 RAG 检索查询，并记录耗时、触发原因和错误。"""

    errors: list[str] = []
    rewrite_started_at_ms = now_ms()
    has_context = bool(summary or len(messages) > 1)
    classification = classification or classify_rag_query(
        original_query=message,
        rewritten_query=message,
        has_context=has_context,
    )

    needs_llm, trigger, skipped_reason = should_llm_rewrite(
        classification=classification,
        has_context=has_context,
    )
    if not needs_llm:
        return RewriteResult(
            query=normalize_query_mark(message),
            errors=errors,
            timing_ms=round(now_ms() - rewrite_started_at_ms, 2),
            mode=REWRITE_MODE_SKIP,
            skipped_reason=skipped_reason,
        )

    try:
        rewritten = rewrite_query(
            build_rewrite_messages(messages=messages, summary=summary),
            profile=PROFILE_REWRITE,
        )
    except Exception as exc:
        rewritten = normalize_query_mark(message)
        errors.append(build_error_info(exc, stage="rewrite_query", source="llm"))

    return RewriteResult(
        query=normalize_query_mark(rewritten),
        errors=errors,
        timing_ms=round(now_ms() - rewrite_started_at_ms, 2),
        mode=REWRITE_MODE_LLM,
        trigger=trigger,
    )

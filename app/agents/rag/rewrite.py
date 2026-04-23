"""rag_agent 查询改写辅助函数。"""

from app.agents.rag.types import RewriteResult
from app.constants.model_profiles import PROFILE_REWRITE
from app.llm import rewrite_query
from app.utils.errors import build_error_info
from app.utils.logger import now_ms


def simple_rewrite(message: str) -> str | None:
    """对短句和“那 xx 呢”这类追问做确定性轻量改写。"""

    message = message.strip()

    if message.endswith("呢"):
        city = message.replace("那", "").replace("呢", "").strip()
        if city:
            return f"{city}气候怎么样？"

    if len(message) <= 4 and not message.endswith("？"):
        return f"{message}气候怎么样？"

    return None


def get_user_messages(messages: list[dict], limit: int = 2) -> list[dict]:
    """取最近几条用户消息，供 LLM rewrite 补充上下文。"""

    user_msgs = [m for m in messages if m["role"] == "user"]
    return user_msgs[-limit:]


def rewrite_rag_query(
    message: str,
    *,
    messages: list[dict],
    summary: str,
) -> RewriteResult:
    """改写 RAG 检索查询，并记录耗时和错误。"""

    errors: list[str] = []
    rewrite_started_at_ms = now_ms()

    rewritten = simple_rewrite(message)
    if not rewritten:
        if message.startswith("那"):
            rewrite_input = []
            if summary:
                rewrite_input.append(
                    {"role": "system", "content": f"对话背景：{summary}"}
                )
            rewrite_input.extend(get_user_messages(messages, 2))
            try:
                rewritten = rewrite_query(rewrite_input, profile=PROFILE_REWRITE)
            except Exception as exc:
                rewritten = message if message.endswith("？") else f"{message}？"
                errors.append(
                    build_error_info(exc, stage="rewrite_query", source="llm")
                )
        else:
            rewritten = message if message.endswith("？") else f"{message}？"

    return RewriteResult(
        query=rewritten,
        errors=errors,
        timing_ms=round(now_ms() - rewrite_started_at_ms, 2),
    )

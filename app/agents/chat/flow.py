from app.agents.chat.answer_strategies import (
    answer_existence_query,
    answer_fallback_summary,
    answer_memory_query,
    answer_summary_query,
)
from app.agents.chat.intent import (
    OPERATOR_EXISTENCE,
    TASK_SUMMARY,
)
from app.agents.chat.types import ChatAnswerResult
from app.constants.policies import MEMORY_POLICY_WORKING_ONLY
from app.utils.logger import now_ms


def generate_chat_answer(
    *,
    operator: str,
    task: str,
    memory_lookup_policy: str,
    memory_hits: list[dict],
    message: str,
    messages: list[dict],
    session_id: str,
    summary: str,
    conversation_history_path: str,
    memory_context: str,
    facts_text: str,
    on_delta=None,
) -> ChatAnswerResult:
    """根据 operator/task/memory 状态选择具体回答策略。"""

    started_at_ms = now_ms()

    if operator == OPERATOR_EXISTENCE:
        result = answer_existence_query(message=message, session_id=session_id)
    elif task == TASK_SUMMARY and memory_lookup_policy == MEMORY_POLICY_WORKING_ONLY:
        result = answer_summary_query(
            message=message,
            messages=messages,
            session_id=session_id,
            summary=summary,
            conversation_history_path=conversation_history_path,
            on_delta=on_delta,
        )
    elif memory_hits:
        result = answer_memory_query(
            message=message,
            summary=summary,
            memory_context=memory_context,
            facts_text=facts_text,
            task=task,
            on_delta=on_delta,
        )
    else:
        result = answer_fallback_summary(
            message=message,
            summary=summary,
            on_delta=on_delta,
        )

    result.sub_timings_ms["answerGeneration"] = round(now_ms() - started_at_ms, 2)
    return result

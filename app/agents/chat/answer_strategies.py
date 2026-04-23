from app.agents.chat.answer import generate_answer, generate_summary_answer
from app.agents.chat.history_summary import (
    SUMMARY_HEADING_HISTORY,
    SUMMARY_HEADING_RECENT,
    build_recent_user_question_items,
    build_summary_items_from_history,
    generate_summary_from_items,
    get_summary_history_events,
)
from app.agents.chat.intent import extract_city_from_query, is_immediate_summary_query
from app.agents.chat.types import ChatAnswerResult
from app.constants.policies import HISTORY_POLICY_NONE, HISTORY_POLICY_RECENT
from app.llm import LLMCallError
from app.memory.vector_memory import build_global_memory_index
from app.utils.errors import build_error_info
from app.utils.logger import now_ms


def answer_existence_query(
    *,
    message: str,
    session_id: str,
) -> ChatAnswerResult:
    """回答“有没有查过/是否问过”这类存在性问题。"""

    started_at_ms = now_ms()
    errors: list[str] = []
    try:
        global_index = build_global_memory_index(session_id=session_id)
    except Exception as exc:
        global_index = {"cities": []}
        errors.append(
            build_error_info(
                exc,
                stage="build_global_memory_index",
                source="memory",
                preferred_code="retrieval_error",
            )
        )

    city = extract_city_from_query(message)
    if city:
        if city in global_index["cities"]:
            answer = f"查询过{city}的相关信息。"
        else:
            answer = f"没有查询过{city}的相关信息。"
    else:
        answer = "资料不足"

    return ChatAnswerResult(
        answer=answer,
        used_memory=True,
        errors=errors,
        sub_timings_ms={
            "existenceLookup": round(now_ms() - started_at_ms, 2),
        },
    )


def answer_summary_query(
    *,
    message: str,
    messages: list[dict],
    session_id: str,
    summary: str,
    conversation_history_path: str,
    on_delta=None,
) -> ChatAnswerResult:
    """回答顺序型 summary 问题，优先 Working Memory，再查 conversation history。"""

    errors: list[str] = []
    working_memory_items: list[str] = []
    history_events: list[dict] = []
    history_items: list[str] = []
    history_lookup_policy = HISTORY_POLICY_NONE
    used_history = False
    used_summary = False

    if is_immediate_summary_query(message):
        working_memory_items = build_recent_user_question_items(
            messages,
            message,
            limit=5,
        )

    if is_immediate_summary_query(message) and working_memory_items:
        answer = generate_summary_from_items(working_memory_items)
    else:
        history_events, history_lookup_policy = get_summary_history_events(
            message,
            session_id,
            conversation_history_path,
        )
        history_items = build_summary_items_from_history(history_events)
        heading = (
            SUMMARY_HEADING_RECENT
            if history_lookup_policy == HISTORY_POLICY_RECENT
            else SUMMARY_HEADING_HISTORY
        )
        answer = generate_summary_from_items(history_items, heading=heading)
        used_history = bool(history_items)

        if not history_items and summary:
            try:
                answer = generate_summary_answer(message, summary, on_delta=on_delta)
                used_summary = True
            except LLMCallError as exc:
                answer = "资料不足"
                errors.append(
                    build_error_info(
                        exc,
                        stage="generate_summary_answer",
                        source="llm",
                    )
                )

    return ChatAnswerResult(
        answer=answer,
        history_lookup_policy=history_lookup_policy,
        working_memory_items=working_memory_items,
        history_events=history_events,
        history_items=history_items,
        used_memory=bool(working_memory_items or history_items),
        used_history=used_history,
        used_summary=used_summary,
        errors=errors,
        sub_timings_ms={"existenceLookup": 0.0},
    )


def answer_memory_query(
    *,
    message: str,
    summary: str,
    memory_context: str,
    facts_text: str,
    task: str,
    on_delta=None,
) -> ChatAnswerResult:
    """基于 Chroma memory 命中生成回答。"""

    errors: list[str] = []
    try:
        answer = generate_answer(
            message,
            summary,
            memory_context,
            facts_text,
            task,
            on_delta=on_delta,
        )
    except LLMCallError as exc:
        answer = "记忆问答暂时失败，请稍后再试。"
        errors.append(build_error_info(exc, stage="generate_answer", source="llm"))

    return ChatAnswerResult(
        answer=answer,
        used_memory=True,
        used_summary=bool(summary),
        errors=errors,
        sub_timings_ms={"existenceLookup": 0.0},
    )


def answer_fallback_summary(
    *,
    message: str,
    summary: str,
    on_delta=None,
) -> ChatAnswerResult:
    """没有可用 memory/history 时，尝试基于 summary 做兜底回答。"""

    errors: list[str] = []
    try:
        answer = generate_summary_answer(message, summary, on_delta=on_delta)
    except LLMCallError as exc:
        answer = "资料不足"
        errors.append(build_error_info(exc, stage="generate_summary_answer", source="llm"))

    return ChatAnswerResult(
        answer=answer,
        used_memory=False,
        used_summary=bool(summary),
        errors=errors,
        sub_timings_ms={"existenceLookup": 0.0},
    )

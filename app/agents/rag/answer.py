"""RAG 回答生成。"""

from app.config import RAG_CONFIG
from app.constants.model_profiles import PROFILE_DEFAULT_CHAT
from app.llm import chat
from app.agents.rag.types import RagAnswerResult, RagContext
from app.prompts.rag import (
    RAG_MEMORY_ANSWER_SYSTEM_PROMPT,
    build_rag_doc_answer_system_prompt,
    build_rag_doc_answer_user_prompt,
    build_rag_memory_answer_user_prompt,
)
from app.utils.errors import build_error_info
from app.utils.logger import now_ms


def generate_doc_answer(
    question: str,
    context: str,
    strategy: dict,
    on_delta=None,
) -> str:
    """基于知识库上下文生成回答。"""

    return chat(
        [
            {
                "role": "system",
                "content": build_rag_doc_answer_system_prompt(strategy["answer_style"]),
            },
            {
                "role": "user",
                "content": build_rag_doc_answer_user_prompt(question, context),
            },
        ],
        max_completion_tokens=strategy["max_tokens"],
        on_delta=on_delta,
        profile=PROFILE_DEFAULT_CHAT,
    )


def generate_memory_answer(question: str, memory_context: str, on_delta=None) -> str:
    """基于历史 memory 上下文生成回答。"""

    return chat(
        [
            {
                "role": "system",
                "content": RAG_MEMORY_ANSWER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_rag_memory_answer_user_prompt(
                    question, memory_context
                ),
            },
        ],
        max_completion_tokens=RAG_CONFIG.max_memory_answer_tokens,
        on_delta=on_delta,
        profile=PROFILE_DEFAULT_CHAT,
    )


def generate_answer_for_context(
    *,
    question: str,
    rag_context: RagContext,
    doc_answer_strategy: dict,
    has_strong_knowledge: bool,
    has_memory: bool,
    on_delta=None,
) -> RagAnswerResult:
    """根据 RAG 上下文选择 doc/memory/兜底回答分支。"""

    errors: list[str] = []
    answer_generation_started_at_ms = now_ms()

    try:
        if has_strong_knowledge:
            answer = generate_doc_answer(
                question,
                rag_context.context,
                doc_answer_strategy,
                on_delta=on_delta,
            )
        elif has_memory:
            answer = generate_memory_answer(
                question,
                rag_context.memory_context,
                on_delta=on_delta,
            )
        else:
            answer = "资料不足"
    except Exception as exc:
        answer = "知识检索暂时失败，请稍后再试。"
        errors.append(build_error_info(exc, stage="answer_generation", source="llm"))

    return RagAnswerResult(
        answer=answer,
        errors=errors,
        timing_ms=round(now_ms() - answer_generation_started_at_ms, 2),
    )

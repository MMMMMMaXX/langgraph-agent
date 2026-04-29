"""RAG 回答生成。"""

from app.agents.rag.constants import ANSWER_TOKENS_CONTEXT_RATIO, ANSWER_TOKENS_MIN
from app.agents.rag.types import RagAnswerResult, RagContext
from app.config import RAG_CONFIG
from app.constants.model_profiles import PROFILE_DEFAULT_CHAT
from app.llm import chat
from app.prompts.rag import (
    RAG_MEMORY_ANSWER_SYSTEM_PROMPT,
    build_citation_correction_user_prompt,
    build_rag_doc_answer_system_prompt,
    build_rag_doc_answer_user_prompt,
    build_rag_memory_answer_user_prompt,
)
from app.utils.errors import build_error_info
from app.utils.logger import now_ms


def check_citation_coverage(answer: str, citations: list[dict]) -> bool:
    """检查答案中是否包含至少一个有效引用标记 [N]。

    citations 为空时（无文档命中）直接返回 True，避免误报。
    只要有一个 ref 存在就视为通过，不要求全部覆盖。
    """

    if not citations:
        return True
    return any(citation["ref"] in answer for citation in citations)


def _correct_missing_citations(
    *,
    question: str,
    context: str,
    answer: str,
    citations: list[dict],
    strategy: dict,
) -> str | None:
    """对缺少引用的答案做一次 self-correction，失败时返回 None。

    采用多轮 continuation：把原始答案作为 assistant 上文，
    在新的 user turn 里要求补引用，保持措辞不变。
    """

    refs = [c["ref"] for c in citations]
    try:
        corrected = chat(
            [
                {
                    "role": "system",
                    "content": build_rag_doc_answer_system_prompt(
                        strategy["answer_style"]
                    ),
                },
                {
                    "role": "user",
                    "content": build_rag_doc_answer_user_prompt(question, context),
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
                {
                    "role": "user",
                    "content": build_citation_correction_user_prompt(answer, refs),
                },
            ],
            max_completion_tokens=strategy["max_tokens"],
            profile=PROFILE_DEFAULT_CHAT,
        )
        return corrected if corrected and corrected.strip() else None
    except Exception:
        return None


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


def generate_memory_answer(
    question: str,
    memory_context: str,
    *,
    max_tokens: int | None = None,
    on_delta=None,
) -> str:
    """基于历史 memory 上下文生成回答。

    max_tokens 由调用方根据实际 memory_context 长度动态传入；
    未传时退回到 RAG_CONFIG.max_memory_answer_tokens。
    """

    effective_max = (
        max_tokens if max_tokens is not None else RAG_CONFIG.max_memory_answer_tokens
    )
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
        max_completion_tokens=effective_max,
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
    citation_correction: dict = {}
    answer_generation_started_at_ms = now_ms()

    try:
        if has_strong_knowledge:
            answer = generate_doc_answer(
                question,
                rag_context.context,
                doc_answer_strategy,
                on_delta=on_delta,
            )

            # ===== Citation 后验校验 =====
            # 非流式模式下：检查是否有引用标记，缺失时触发一次 self-correction。
            # 流式模式下：答案已推给客户端，只记录覆盖率，不做修正。
            if rag_context.citations:
                has_coverage = check_citation_coverage(answer, rag_context.citations)
                citation_correction = {
                    "triggered": not has_coverage,
                    "corrected": False,
                    "coverage": has_coverage,
                    "skip_reason": "streaming" if on_delta is not None else "",
                }
                if not has_coverage and on_delta is None:
                    corrected = _correct_missing_citations(
                        question=question,
                        context=rag_context.context,
                        answer=answer,
                        citations=rag_context.citations,
                        strategy=doc_answer_strategy,
                    )
                    if corrected:
                        answer = corrected
                        citation_correction["corrected"] = True

        elif has_memory:
            memory_max_tokens = max(
                ANSWER_TOKENS_MIN,
                min(
                    RAG_CONFIG.max_memory_answer_tokens,
                    len(rag_context.memory_context) // ANSWER_TOKENS_CONTEXT_RATIO,
                ),
            )
            answer = generate_memory_answer(
                question,
                rag_context.memory_context,
                max_tokens=memory_max_tokens,
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
        citation_correction=citation_correction,
    )

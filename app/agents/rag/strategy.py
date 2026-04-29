"""RAG 回答策略选择。"""

from app.agents.rag.constants import (
    ANSWER_STRATEGY_COMPARISON,
    ANSWER_STRATEGY_FALLBACK,
    ANSWER_STRATEGY_FOLLOWUP,
    ANSWER_TOKENS_CONTEXT_RATIO,
    ANSWER_TOKENS_MIN,
    DEFINITION_CONTEXT_CHARS,
    DEFINITION_MAX_ANSWER_TOKENS,
    QUERY_TYPE_COMPARISON,
    QUERY_TYPE_DEFINITION,
    QUERY_TYPE_FALLBACK,
    QUERY_TYPE_FOLLOWUP,
)
from app.agents.rag.types import QueryClassification
from app.config import RAG_CONFIG
from app.constants.policies import (
    ANSWER_STRATEGY_DEFAULT_SHORT,
    ANSWER_STRATEGY_DEFINITION_SHORT,
)


def is_definition_query(message: str) -> bool:
    """判断是否是定义类问题。"""

    return any(keyword in message for keyword in ["是什么", "什么是", "定义", "概念"])


def adapt_strategy_max_tokens(strategy: dict, actual_context_chars: int) -> dict:
    """根据实际上下文字符数动态收紧 max_tokens。

    strategy 的初始 max_tokens 是策略维度的理论上限；这里进一步根据真实
    context 长度向下约束：短 context 不需要也不应该允许长答案。

    公式：max(ANSWER_TOKENS_MIN, min(策略上限, context_chars // RATIO))
    - 360 chars context → min(180, 180) = 180（正常场景不受影响）
    - 100 chars context → min(180, 50) = 50（短 context 收紧）
    - 30 chars context  → max(40, 15) = 40（极短 context 保底）
    """

    base = strategy["max_tokens"]
    dynamic = max(
        ANSWER_TOKENS_MIN,
        min(base, actual_context_chars // ANSWER_TOKENS_CONTEXT_RATIO),
    )
    if dynamic == base:
        return strategy
    return {**strategy, "max_tokens": dynamic}


def build_doc_answer_strategy(
    question: str,
    classification: QueryClassification | None = None,
) -> dict:
    """根据问题类型选择生成策略。

    定义类问题通常只需要“概念定义 + 关键作用”，不需要长上下文和长答案。
    把策略显式化后，debug 可以直接解释本次生成为什么快/慢、为什么短/长。
    """

    query_type = classification.query_type if classification else ""

    if query_type == QUERY_TYPE_DEFINITION or (
        not classification and is_definition_query(question)
    ):
        return {
            "name": ANSWER_STRATEGY_DEFINITION_SHORT,
            "answer_style": "用不超过2句话回答：第1句直接下定义，第2句只补充最关键作用。",
            "context_chars": DEFINITION_CONTEXT_CHARS,
            "max_tokens": min(
                RAG_CONFIG.max_doc_answer_tokens,
                DEFINITION_MAX_ANSWER_TOKENS,
            ),
        }

    if query_type == QUERY_TYPE_COMPARISON:
        return {
            "name": ANSWER_STRATEGY_COMPARISON,
            "answer_style": "用对比方式回答：先给结论，再列出关键差异；只基于资料，不确定就说明资料不足。",
            "context_chars": RAG_CONFIG.max_doc_context_chars,
            "max_tokens": RAG_CONFIG.max_doc_answer_tokens,
        }

    if query_type == QUERY_TYPE_FOLLOWUP:
        return {
            "name": ANSWER_STRATEGY_FOLLOWUP,
            "answer_style": "结合改写后的完整问题简洁回答，避免复述无关历史；资料不足时直接说明。",
            "context_chars": RAG_CONFIG.max_doc_context_chars,
            "max_tokens": RAG_CONFIG.max_doc_answer_tokens,
        }

    if query_type == QUERY_TYPE_FALLBACK:
        return {
            "name": ANSWER_STRATEGY_FALLBACK,
            "answer_style": "如果资料不足或问题指代不清，不要猜测；简短说明需要更明确的问题或更多资料。",
            "context_chars": DEFINITION_CONTEXT_CHARS,
            "max_tokens": min(
                RAG_CONFIG.max_doc_answer_tokens, DEFINITION_MAX_ANSWER_TOKENS
            ),
        }

    return {
        "name": ANSWER_STRATEGY_DEFAULT_SHORT,
        "answer_style": "优先简洁回答，控制在3句话内，只保留最关键的信息。",
        "context_chars": RAG_CONFIG.max_doc_context_chars,
        "max_tokens": RAG_CONFIG.max_doc_answer_tokens,
    }

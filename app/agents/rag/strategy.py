"""RAG 回答策略选择。"""

from app.agents.rag.constants import (
    DEFINITION_CONTEXT_CHARS,
    DEFINITION_MAX_ANSWER_TOKENS,
)
from app.config import RAG_CONFIG
from app.constants.policies import (
    ANSWER_STRATEGY_DEFAULT_SHORT,
    ANSWER_STRATEGY_DEFINITION_SHORT,
)


def is_definition_query(message: str) -> bool:
    """判断是否是定义类问题。"""

    return any(keyword in message for keyword in ["是什么", "什么是", "定义", "概念"])


def build_doc_answer_strategy(question: str) -> dict:
    """根据问题类型选择生成策略。

    定义类问题通常只需要“概念定义 + 关键作用”，不需要长上下文和长答案。
    把策略显式化后，debug 可以直接解释本次生成为什么快/慢、为什么短/长。
    """

    if is_definition_query(question):
        return {
            "name": ANSWER_STRATEGY_DEFINITION_SHORT,
            "answer_style": "用不超过2句话回答：第1句直接下定义，第2句只补充最关键作用。",
            "context_chars": DEFINITION_CONTEXT_CHARS,
            "max_tokens": min(
                RAG_CONFIG.max_doc_answer_tokens,
                DEFINITION_MAX_ANSWER_TOKENS,
            ),
        }

    return {
        "name": ANSWER_STRATEGY_DEFAULT_SHORT,
        "answer_style": "优先简洁回答，控制在3句话内，只保留最关键的信息。",
        "context_chars": RAG_CONFIG.max_doc_context_chars,
        "max_tokens": RAG_CONFIG.max_doc_answer_tokens,
    }


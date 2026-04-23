from app.constants.keywords import (
    EXISTENCE_QUERY_KEYWORDS,
    IMMEDIATE_SUMMARY_QUERY_KEYWORDS,
    RECALL_QUERY_KEYWORDS,
    SUMMARY_QUERY_KEYWORDS,
    contains_any,
)
from app.constants.tags import CITY_TAGS
from app.agents.chat.constants import (
    OPERATOR_AGGREGATE,
    OPERATOR_EXISTENCE,
    OPERATOR_LOOKUP,
    TASK_QA,
    TASK_RECALL,
    TASK_SUMMARY,
)


def classify_chat_task(message: str) -> str:
    """把 chat 请求粗分为 summary / recall / qa。"""

    if contains_any(message, SUMMARY_QUERY_KEYWORDS):
        return TASK_SUMMARY
    if contains_any(message, RECALL_QUERY_KEYWORDS):
        return TASK_RECALL
    return TASK_QA


def classify_chat_operator(message: str) -> str:
    """把 chat 请求转换成更细的操作符，供主流程选择执行分支。"""

    if contains_any(message, SUMMARY_QUERY_KEYWORDS):
        return OPERATOR_AGGREGATE
    if contains_any(message, EXISTENCE_QUERY_KEYWORDS):
        return OPERATOR_EXISTENCE
    if contains_any(message, RECALL_QUERY_KEYWORDS):
        return OPERATOR_LOOKUP
    return OPERATOR_LOOKUP


def is_immediate_summary_query(message: str) -> bool:
    """判断用户是否在问“刚才/刚刚”这种强时效总结。

    这类问题应该只依赖当前进程内的 Working Memory。服务重启后，
    Working Memory 会清空，此时继续回退到 Chroma 长期记忆会把很久以前的
    session 历史误当成“刚才”，所以需要显式截断 fallback。
    """

    return classify_chat_task(message) == TASK_SUMMARY and contains_any(
        message, IMMEDIATE_SUMMARY_QUERY_KEYWORDS
    )


def extract_city_from_query(message: str) -> str | None:
    """从用户问题中提取当前支持的城市标签。"""

    for city in CITY_TAGS:
        if city in message:
            return city
    return None

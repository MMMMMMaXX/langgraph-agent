from app.agents.chat.intent import (
    TASK_SUMMARY,
    classify_chat_task,
    is_immediate_summary_query,
)
from app.config import CONVERSATION_HISTORY_CONFIG
from app.constants.policies import (
    HISTORY_POLICY_ALL,
    HISTORY_POLICY_RECENT,
    INSUFFICIENT_KNOWLEDGE_ANSWER,
)
from app.memory.conversation_history import get_all_history, get_recent_history

# 最近问题总结标题：用于“刚才/刚刚”这类强时效总结。
SUMMARY_HEADING_RECENT = "刚刚的问题包括："

# 全量历史总结标题：用于“总结所有问题/历史问题”。
SUMMARY_HEADING_HISTORY = "历史问题包括："


def extract_question_from_history_event(event: dict) -> str:
    """从会话流水事件中提取用于展示的问题文本。"""

    question = str(event.get("rewritten_query") or event.get("user_message") or "")
    return question.strip().rstrip("？?")


def build_summary_items_from_history(events: list[dict]) -> list[str]:
    """从非向量化会话流水中提取问题列表。

    会话流水天然按时间顺序保存，比向量检索更适合回答“刚才/所有问题”这类顺序型总结。
    """

    items: list[str] = []
    ordered_events = sorted(events, key=lambda item: item.get("timestamp", 0))
    for event in ordered_events:
        question = extract_question_from_history_event(event)
        if not question:
            continue
        if classify_chat_task(question) == TASK_SUMMARY:
            continue
        if question in items:
            continue
        items.append(question)
    return items


def build_recent_user_question_items(
    messages: list[dict],
    current_message: str,
    limit: int = 5,
) -> list[str]:
    """从 Working Memory 里提取最近用户问题。

    对“刚才/刚刚”的总结来说，当前进程里的 messages 比长期向量记忆更可信：
    - messages 表示这次 session 真实连续发生的对话
    - Chroma memory 表示长期事实库，可能包含很久以前迁移进来的历史
    """

    items: list[str] = []
    current_message = current_message.strip()

    for item in reversed(messages):
        if item.get("role") != "user":
            continue
        content = str(item.get("content", "")).strip()
        if not content or content == current_message:
            continue
        if classify_chat_task(content) == TASK_SUMMARY:
            continue
        content = content.rstrip("？?")
        if content in items:
            continue
        items.append(content)
        if len(items) >= limit:
            break

    return list(reversed(items))


def generate_summary_from_items(
    items: list[str],
    heading: str = SUMMARY_HEADING_RECENT,
) -> str:
    """把问题列表渲染成用户可读的总结回答。"""

    if not items:
        return INSUFFICIENT_KNOWLEDGE_ANSWER

    lines = [heading]
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. 用户询问{item}")
    return "\n".join(lines)


def get_summary_history_events(
    message: str,
    session_id: str,
    history_path: str = "",
) -> tuple[list[dict], str]:
    """根据 summary 类型读取 recent 或 all conversation history。"""

    if not is_immediate_summary_query(message):
        return (
            get_all_history(
                session_id=session_id,
                limit=CONVERSATION_HISTORY_CONFIG.all_limit,
                history_path=history_path,
            ),
            HISTORY_POLICY_ALL,
        )

    return (
        get_recent_history(
            session_id=session_id,
            limit=CONVERSATION_HISTORY_CONFIG.recent_limit,
            history_path=history_path,
        ),
        HISTORY_POLICY_RECENT,
    )

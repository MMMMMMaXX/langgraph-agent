from app.constants.model_profiles import PROFILE_DEFAULT_CHAT, PROFILE_SUMMARY
from app.llm import chat
from app.prompts.chat import (
    CHAT_SUMMARY_SYSTEM_PROMPT,
    build_chat_qa_system_prompt,
)


def generate_answer(
    message: str,
    summary: str,
    memory_context: str,
    facts_text: str,
    task: str,
    on_delta=None,
) -> str:
    """基于 memory context 和结构化 facts 生成普通 chat/memory 问答。"""

    return chat(
        [
            {
                "role": "system",
                "content": build_chat_qa_system_prompt(task),
            },
            {
                "role": "user",
                "content": f"""
历史记忆（原始）：
{memory_context or "无"}

{facts_text}

历史摘要（辅助）：
{summary or "无"}

问题：
{message}
""".strip(),
            },
        ],
        on_delta=on_delta,
        profile=PROFILE_DEFAULT_CHAT,
    )


def generate_summary_answer(message: str, summary: str, on_delta=None) -> str:
    """在没有直接 history items 时，基于 summary 生成兜底总结回答。"""

    return chat(
        [
            {
                "role": "system",
                "content": CHAT_SUMMARY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
历史摘要：
{summary or "无"}

问题：
{message}
""".strip(),
            },
        ],
        on_delta=on_delta,
        profile=PROFILE_SUMMARY,
    )

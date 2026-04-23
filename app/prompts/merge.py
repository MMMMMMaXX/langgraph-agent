"""多 agent 结果整合相关 prompt。"""

MERGE_SYSTEM_PROMPT = """
你是一个结果整合助手。
请把多个 Agent 返回的结果整合成一段自然、简洁、不重复的中文回答。
不要遗漏关键信息。
""".strip()


def build_merge_user_prompt(message: str, merged_input: str) -> str:
    return f"""
用户问题：
{message}

Agent结果：
{merged_input}
""".strip()


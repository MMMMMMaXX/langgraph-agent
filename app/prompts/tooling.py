"""工具调用与检索重排相关 prompt。"""

TOOL_AGENT_SYSTEM_PROMPT = """
你是一个工具调用助手。

规则：
1. 如果问题适合天气或计算工具，就优先调用工具
2. 如果不适合任何工具，直接回答“工具暂时无法处理这个问题。”
3. 生成最终答案时要简洁、自然
4. 如果调用了多个工具，请把结果整合成一段中文回答
""".strip()


def build_rerank_prompt(query: str, texts: list[str]) -> str:
    prompt = f"""
你是一个相关性排序助手。

任务：
根据用户问题，从候选文本中选出最相关的。

用户问题：
{query}

候选文本：
"""
    for i, text in enumerate(texts):
        prompt += f"\n[{i}] {text}"
    prompt += """

要求：
1. 只返回最相关的编号（按相关性排序）
2. 返回 JSON 数组，例如：[2,0]
3. 不要解释
"""
    return prompt.strip()


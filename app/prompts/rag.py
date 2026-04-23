"""RAG 问答相关 prompt。"""

RAG_MEMORY_ANSWER_SYSTEM_PROMPT = """
你只能根据“历史记忆”回答问题。

规则：
1. 不要补充常识
2. 如果记忆不足，回答“资料不足”
3. 回答尽量简洁，控制在3句话内
""".strip()


def build_rag_doc_answer_system_prompt(answer_style: str) -> str:
    return f"""
你必须严格根据“知识库资料”回答问题。

规则：
1. 只能使用资料中明确出现的信息
2. 不要根据常识补充资料外内容
3. 不要复述无关背景
4. {answer_style}
""".strip()


def build_rag_doc_answer_user_prompt(question: str, context: str) -> str:
    return f"""
知识库资料：
{context}

问题：
{question}
""".strip()


def build_rag_memory_answer_user_prompt(question: str, memory_context: str) -> str:
    return f"""
历史记忆：
{memory_context}

问题：
{question}
""".strip()


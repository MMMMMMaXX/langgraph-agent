"""RAG 问答相关 prompt。"""

from app.constants.policies import INSUFFICIENT_KNOWLEDGE_ANSWER

RAG_MEMORY_ANSWER_SYSTEM_PROMPT = f"""
你只能根据"历史记忆"回答问题。

规则：
1. 不要补充常识
2. 如果记忆不足，回答"{INSUFFICIENT_KNOWLEDGE_ANSWER}"
3. 回答尽量简洁，控制在3句话内
""".strip()


def build_rag_doc_answer_system_prompt(answer_style: str) -> str:
    return f"""
你必须严格根据"知识库资料"回答问题。

规则：
1. 只能使用资料中明确出现的信息
2. 不要根据常识补充资料外内容
3. 不要复述无关背景
4. 如果资料中带有 [1]、[2] 这类引用编号，请在关键结论后保留对应编号
5. {answer_style}
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


def build_citation_correction_user_prompt(answer: str, refs: list[str]) -> str:
    """引用修正 prompt：要求 LLM 在已有答案基础上补全引用标记。

    采用多轮 continuation 形式，把原答案作为 assistant 上文，
    在新的 user turn 里要求补引用，不改措辞。
    """

    refs_str = "、".join(refs)
    return f"""你的回答缺少来源引用标记。请在每个关键结论的句尾补充对应编号（{refs_str}），不要修改答案的措辞或语气。

原始回答：
{answer}

请直接输出补充引用后的完整回答，不要解释。""".strip()


# ---------------------------------------------------------------------------
# LLM Query Classifier prompt
# ---------------------------------------------------------------------------

_CLASSIFIER_TYPES_GUIDE = """\
- definition : 问概念/定义/用法/功能，例：「是什么」「怎么用」「有什么作用」
- comparison : 对比/区别/优缺点，例：「A 和 B 有什么区别」「vs」
- factual    : 具体事实/数据查询，例：「北京今天气温」「Python 最新版本号」
- followup   : 追问，依赖上下文，含代词「它/这/该/那个」，例：「那它呢」
- fallback   : 意图不明或信息严重不足"""

_CLASSIFIER_EXAMPLES = """\
问题: OAuth2 是什么
→ {"type": "definition", "confidence": 0.95}

问题: Spring Boot 3.2 和 3.1 有什么区别
→ {"type": "comparison", "confidence": 0.95}

问题: 服务器当前 CPU 使用率多少
→ {"type": "factual", "confidence": 0.90}

问题: 那它的配置怎么写
→ {"type": "followup", "confidence": 0.88}

问题: Redis 缓存穿透如何解决
→ {"type": "definition", "confidence": 0.88}

问题: 嗯
→ {"type": "fallback", "confidence": 0.90}"""

CLASSIFIER_SYSTEM_PROMPT = f"""你是 query 分类助手。把用户问题归类到唯一类别：
{_CLASSIFIER_TYPES_GUIDE}

只返回 JSON，不要解释，格式：{{"type": "xxx", "confidence": 0.xx}}

示例：
{_CLASSIFIER_EXAMPLES}""".strip()


def build_classifier_user_prompt(query: str) -> str:
    return f"问题: {query.strip()}"

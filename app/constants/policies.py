# 定义类短回答策略：用于“X 是什么”这类知识定义问答，控制上下文和回答长度。
ANSWER_STRATEGY_DEFINITION_SHORT = "definition_short"

# 默认短回答策略：用于普通 RAG 问答的短答案生成。
ANSWER_STRATEGY_DEFAULT_SHORT = "default_short"

# 只使用 Working Memory：通常用于“刚才/刚刚总结”，避免误捞长期旧记忆。
MEMORY_POLICY_WORKING_ONLY = "working_memory_only"

# 使用长期语义记忆：通常用于普通回忆或基于 memory 的问答。
MEMORY_POLICY_SEMANTIC_LONG_TERM = "semantic_long_term_memory"

# 不读取会话流水：当前问题不需要 history 辅助。
HISTORY_POLICY_NONE = "none"

# 读取最近会话流水：用于强时效问题，如“刚才问了什么”。
HISTORY_POLICY_RECENT = "recent_history"

# 读取全部会话流水：用于“总结所有问题/历史问题”。
HISTORY_POLICY_ALL = "all_history"

# 跳过原因：答案为空，不能写入 memory/history。
SKIP_REASON_EMPTY_ANSWER = "empty_answer"

# 跳过原因：答案质量不适合持久化，例如“资料不足”或无法处理。
SKIP_REASON_BAD_ANSWER = "bad_answer"

# 跳过原因：答案过短，信息密度不足。
SKIP_REASON_TOO_SHORT = "too_short"

# 跳过原因：用户问的是总结/回顾/是否问过等元问题，避免污染长期记忆。
SKIP_REASON_META_QUERY = "meta_query"

# 跳过原因：工具调用结果通常是短期信息，不默认进入长期语义记忆。
SKIP_REASON_TOOL_REQUEST = "tool_request"

# 跳过原因：创作输出更适合保存成产物，不整段写入 Chroma memory。
SKIP_REASON_CREATIVE_OUTPUT = "creative_output"

# 跳过原因：RAG 已命中文档，可从 docs 再现，不重复写入 vector memory。
SKIP_REASON_RAG_DOC_HIT = "rag_doc_hit"

# 兼容旧调试字段：历史 eval / history 里可能已有这个 skip reason。
SKIP_REASON_RAG_DEFINITION_DOC_HIT = "rag_definition_doc_hit"

# 跳过原因：RAG 已命中文档，因此 memory 检索可以跳过。
SKIP_REASON_DOC_HIT = "doc_hit"

# 跳过原因：用户消息为空，不写入会话流水。
SKIP_REASON_EMPTY_MESSAGE = "empty_message"

# 跳过原因：会话流水检测到短时间重复事件，避免重复记录。
SKIP_REASON_DUPLICATE = "duplicate"

# RAG rerank 跳过原因：候选只有一条，没有必要 rerank。
DOC_RERANK_SKIP_REASON_SINGLE_CANDIDATE = "single_candidate"

# RAG rerank 跳过原因：相邻 chunk 高置信命中，直接合并更稳定且省一次 LLM 调用。
DOC_RERANK_SKIP_REASON_ADJACENT_HIGH_CONFIDENCE = "adjacent_high_confidence"

# RAG 知识不足时返回给用户的兜底答案：answer.py 生成，write_policy.py 检测。
# 两处共享同一字符串，避免写入了不该写的"知识不足"兜底答案。
INSUFFICIENT_KNOWLEDGE_ANSWER = "资料不足"

# 表示本次 LLM 调用无法处理请求的信号词，用于识别低质量答案，阻止其写入长期 memory。
# write_policy.py（写入决策）和 chat/memory_retrieval.py（检索过滤）共用此集合。
BAD_ANSWER_KEYWORDS = ("无法处理",)

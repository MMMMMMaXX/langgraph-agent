"""chat agent 的任务类型、操作符和检索阈值。"""

# 总结任务：聚合历史问题或对话内容。
TASK_SUMMARY = "summary"

# 回忆任务：查询刚刚、刚才、之前是否发生过某件事。
TASK_RECALL = "recall"

# 普通问答任务：基于已有 memory 或 summary 回答用户问题。
TASK_QA = "qa"

# 聚合操作：通常对应“总结/回顾/列出”。
OPERATOR_AGGREGATE = "aggregate"

# 存在性操作：通常对应“有没有/是否/查过/问过”。
OPERATOR_EXISTENCE = "existence"

# 查找操作：普通回忆或 memory lookup。
OPERATOR_LOOKUP = "lookup"

# 普通 chat 语义记忆检索候选数。
CHAT_MEMORY_SEARCH_TOP_K = 8

# 回忆类问题 rerank 后最多消费的事实记忆数。
CHAT_RECALL_RERANK_TOP_K = 3

# 普通问答 rerank 后最多消费的语义记忆数。
CHAT_QA_RERANK_TOP_K = 3

# 低价值 memory 内容长度阈值：过短且无结构信息时不进入回答上下文。
LOW_VALUE_MEMORY_MIN_CONTENT_CHARS = 15

# 低价值 memory 标记：工具/失败类回答不适合作为长期回答上下文。
LOW_VALUE_MEMORY_BLOCK_KEYWORDS = ("无法处理",)

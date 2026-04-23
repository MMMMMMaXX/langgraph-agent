# 普通对话/记忆问答节点：处理闲聊、总结、回忆、基于 memory 的问答。
ROUTE_CHAT_AGENT = "chat_agent"

# 文档 RAG 节点：处理知识库检索、文档问答。
ROUTE_RAG_AGENT = "rag_agent"

# 工具节点：处理天气、计算等外部工具调用。
ROUTE_TOOL_AGENT = "tool_agent"

# 创作改编节点：处理小说转剧本、短剧脚本等长创作任务。
ROUTE_NOVEL_SCRIPT_AGENT = "novel_script_agent"

# Supervisor 节点：负责根据用户输入决定后续走哪些 agent。
NODE_SUPERVISOR = "supervisor"

# Merge 节点：负责合并一个或多个 agent 的输出，产出最终回答。
NODE_MERGE = "merge"

# Memory 节点：负责摘要刷新、向量记忆写入、会话流水写入。
NODE_MEMORY = "memory"

# 允许出现在 state["routes"] 里的 agent 路由集合。
AGENT_ROUTES = (
    ROUTE_TOOL_AGENT,
    ROUTE_RAG_AGENT,
    ROUTE_CHAT_AGENT,
    ROUTE_NOVEL_SCRIPT_AGENT,
)

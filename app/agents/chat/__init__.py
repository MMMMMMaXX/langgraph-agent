"""chat agent 内部模块。

这个包用于逐步拆分原本过长的 `app.agents.chat_agent`：
- intent：负责意图/操作符分类
- policies：负责 memory/history 读取策略

外部 LangGraph 入口仍可通过 `app.agents.chat_agent.chat_agent_node` 访问，
由兼容文件转发到 `app.agents.chat.node.chat_agent_node`。
"""

"""rag agent 内部模块。

这个包用于拆分原本过长的 `app.agents.rag_agent`：
- rewrite：负责查询改写
- doc_pipeline：负责文档检索、过滤、rerank 和 chunk 合并
- memory_pipeline：负责 memory 兜底检索
- strategy：负责回答策略选择
- answer：负责最终回答生成

外部 LangGraph 入口仍可通过 `app.agents.rag_agent.rag_agent_node` 访问，
由兼容文件转发到 `app.agents.rag.node.rag_agent_node`。
"""

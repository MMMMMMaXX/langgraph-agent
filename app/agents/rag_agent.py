"""rag agent 兼容入口。

真实实现已经迁移到 `app.agents.rag.node`。这里保留旧 import 路径，
让 `app.graph`、评测脚本和历史工具不需要立刻跟着调整。
"""

from app.agents.rag.node import rag_agent_node

__all__ = ["rag_agent_node"]


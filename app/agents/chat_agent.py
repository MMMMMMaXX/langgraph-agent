"""chat agent 兼容入口。

真实实现已经迁移到 `app.agents.chat.node`。这里保留旧 import 路径，
让 `app.graph` 和历史脚本不需要立刻跟着调整。
"""

from app.agents.chat.node import chat_agent_node

__all__ = ["chat_agent_node"]

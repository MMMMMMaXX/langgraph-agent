"""运行时状态管理入口。

这一层的目标是把 session cache / checkpoint 细节收口到统一门面后面，
让 chat_service / API 层只感知“加载会话态”和“提交会话态”。
"""

from .session_runtime import SessionRuntime
from .snapshot import ConversationSnapshot

__all__ = ["ConversationSnapshot", "SessionRuntime"]

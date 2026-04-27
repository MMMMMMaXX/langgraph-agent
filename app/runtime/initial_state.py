"""初始会话状态工厂。

之所以把 create_initial_state 单独抽出来，是为了避免 session_store 反向依赖
chat_service，给 runtime 层引入循环导入。
"""

from __future__ import annotations

from app.state import AgentState


def create_initial_state(session_id: str = "default") -> AgentState:
    """创建一个最小可运行的初始 AgentState。"""

    return {
        "session_id": session_id,
        "debug": False,
        "messages": [],
        "summary": "",
    }

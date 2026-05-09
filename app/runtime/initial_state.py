"""初始会话状态工厂。

之所以把 create_initial_state 单独抽出来，是为了避免 session_store 反向依赖
chat_service，给 runtime 层引入循环导入。
"""

from __future__ import annotations

from app.auth.context import AuthContext
from app.constants.auth import (
    ANONYMOUS_TENANT_ID,
    ANONYMOUS_USER_ID,
    ROLE_ANONYMOUS,
)
from app.state import AgentState

# 初始态的默认身份：匿名。
# 真实请求进来前 session cache 就可能被访问（见 session_runtime.build_request_state
# 首次命中时用 initial state 回填 cache），因此这里需要给一个合法的 AuthContext。
# 真正的 auth 会在 build_request_state 里被请求身份覆盖，不会污染下游。
_DEFAULT_AUTH = AuthContext(
    tenant_id=ANONYMOUS_TENANT_ID,
    user_id=ANONYMOUS_USER_ID,
    groups=(),
    role=ROLE_ANONYMOUS,
    anonymous=True,
)


def create_initial_state(session_id: str = "default") -> AgentState:
    """创建一个最小可运行的初始 AgentState。"""

    return {
        "session_id": session_id,
        "debug": False,
        "messages": [],
        "summary": "",
        "auth": _DEFAULT_AUTH,
    }

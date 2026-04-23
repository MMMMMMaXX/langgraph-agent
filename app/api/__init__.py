"""API 包门面。

保持 `uvicorn app.api:app`、`from app.api import app, clear_session_store`
这些老用法继续工作。拆分细节见各子模块。
"""

from __future__ import annotations

from .app import app
from .session_store import (
    clear_session_store,
    get_or_create_session_state,
    session_store,
)

__all__ = [
    "app",
    "clear_session_store",
    "get_or_create_session_state",
    # 暴露 session_store 主要给评测脚本做内省，不建议业务代码直接读写。
    "session_store",
]

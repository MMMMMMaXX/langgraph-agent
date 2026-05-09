"""Auth 模块：身份上下文定义与注入。

导出最常用的符号，避免下游到处写 `from app.auth.context import AuthContext`。
"""

from app.auth.context import AuthContext, AuthRole
from app.auth.injection import AnonymousAuthDisabled, build_auth_context

__all__ = [
    "AnonymousAuthDisabled",
    "AuthContext",
    "AuthRole",
    "build_auth_context",
]

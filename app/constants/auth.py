"""Auth 上下文与匿名 fallback 相关常量。

集中在这里的原因：api / runtime / agents / tests 都会引用这些字符串，
分散维护容易出现"role=admin 写两处、漏改一处"这类 bug（参见项目内的
常量收敛原则）。所有新增跨模块共享的 auth 字符串一律加到本文件，不允许
在调用处内联 magic string。
"""

from __future__ import annotations

from typing import Final

# -------- 环境变量名 --------

# 控制"未带 auth 的请求"是否允许走匿名 fallback。默认生产 fail-closed，
# 本地 dev / eval 通过设置 `ALLOW_ANONYMOUS_AUTH=true` 启用。
ALLOW_ANONYMOUS_AUTH_ENV: Final[str] = "ALLOW_ANONYMOUS_AUTH"

# -------- 匿名上下文默认字段 --------

ANONYMOUS_TENANT_ID: Final[str] = "default"
ANONYMOUS_USER_ID: Final[str] = "anonymous"

# -------- 身份角色 --------

ROLE_ANONYMOUS: Final[str] = "anonymous"
ROLE_USER: Final[str] = "user"
ROLE_ADMIN: Final[str] = "admin"
ROLE_SERVICE: Final[str] = "service"

VALID_ROLES: Final[frozenset[str]] = frozenset(
    {ROLE_ANONYMOUS, ROLE_USER, ROLE_ADMIN, ROLE_SERVICE}
)

# 默认角色：显式传 auth 但没传 role 时使用。
DEFAULT_ROLE: Final[str] = ROLE_USER

# -------- 错误码 / 错误文案 --------

# 401：生产模式下未带 auth 的请求。
ERR_UNAUTHORIZED: Final[str] = "unauthorized"

# 403：匿名上下文试图调用 side_effect 工具时返回。
ERR_ANONYMOUS_FORBIDDEN_SIDE_EFFECT: Final[str] = "anonymous_forbidden_side_effect"

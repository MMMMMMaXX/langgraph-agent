"""Auth 上下文注入层：把 HTTP 请求字段翻译成 AuthContext。

职责：
1. 读取 `ChatRequest.auth`（Pydantic 模型）构造 AuthContext。
2. 处理"请求未带 auth"的兼容策略：
   - `ALLOW_ANONYMOUS_AUTH=true`（dev / eval）→ 返回匿名 AuthContext
   - 否则 → 抛 `AnonymousAuthDisabled`，上游翻成 401

故意把"决定走匿名还是拒绝"集中在这里，避免 routes / chat_runner / agents
三处各自判断环境变量，未来加 JWT 验签只需要替换本模块即可。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from app.auth.context import AuthContext
from app.constants.auth import (
    ALLOW_ANONYMOUS_AUTH_ENV,
    ANONYMOUS_TENANT_ID,
    ANONYMOUS_USER_ID,
    DEFAULT_ROLE,
    ROLE_ANONYMOUS,
    VALID_ROLES,
)

if TYPE_CHECKING:
    from app.api.schemas import AuthRequest


class AnonymousAuthDisabled(Exception):
    """未带 auth 且 `ALLOW_ANONYMOUS_AUTH != true` 时抛出。

    单独建一个异常类型是为了让 API 层能翻成 401 而不是 400/500；
    agent / runtime 层不会直接捕获它。
    """


# 只有明确为真值的字符串才算启用匿名 fallback，避免误把 "0"/"false" 当成 True。
_TRUE_LITERALS = frozenset({"1", "true", "yes", "on"})


def _allow_anonymous() -> bool:
    raw = os.getenv(ALLOW_ANONYMOUS_AUTH_ENV, "").strip().lower()
    return raw in _TRUE_LITERALS


def build_auth_context(auth: AuthRequest | None) -> AuthContext:
    """根据请求体里的 auth 字段构造 AuthContext。

    - 有 `auth`：按值直接构造（role 非法 / 必填为空由 AuthContext 校验）。
    - 无 `auth`：受 `ALLOW_ANONYMOUS_AUTH` 开关控制：
      - true  → 返回匿名 AuthContext
      - false → 抛 AnonymousAuthDisabled（API 层翻成 401）
    """

    if auth is None:
        if not _allow_anonymous():
            raise AnonymousAuthDisabled(
                "request missing auth and ALLOW_ANONYMOUS_AUTH is not enabled"
            )
        return AuthContext(
            tenant_id=ANONYMOUS_TENANT_ID,
            user_id=ANONYMOUS_USER_ID,
            groups=(),
            role=ROLE_ANONYMOUS,
            anonymous=True,
        )

    role = (auth.role or DEFAULT_ROLE).strip()
    if role not in VALID_ROLES:
        raise ValueError(f"invalid auth role: {role!r}")

    anonymous = role == ROLE_ANONYMOUS
    return AuthContext(
        tenant_id=auth.tenant_id.strip(),
        user_id=auth.user_id.strip(),
        groups=tuple(g for g in auth.groups if g),
        role=role,  # type: ignore[arg-type]
        anonymous=anonymous,
    )

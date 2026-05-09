"""Auth 上下文：tenant_id / user_id / groups / role 的强类型承载。

本模块只定义"身份是什么"，不负责"如何获取身份"。获取逻辑放在
`app.auth.injection`，用来隔离"从 HTTP schema 翻译成 AuthContext"的
策略（匿名 fallback、401 拒绝等）。

设计要点：
- `frozen=True`：AuthContext 在整个请求链路传递，禁止被 agent 节点改写；
  如需要派生身份（如服务调用降级），显式 `dataclasses.replace` 产生新实例。
- `groups` 用 tuple（而不是 list），保证可哈希、可作为字典 key、在 trace /
  audit 日志里序列化稳定。
- `role=anonymous` 和 `anonymous=True` **必须同时成立**；在构造时做断言，
  避免两处真值漂移。

Phase 1 只覆盖"有身份/匿名"两种状态，permissions/scopes/token_expires_at
等留给 Phase 2+ 扩展。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.constants.auth import ROLE_ANONYMOUS, VALID_ROLES

AuthRole = Literal["anonymous", "user", "admin", "service"]


@dataclass(frozen=True)
class AuthContext:
    """贯穿一次请求的身份上下文。"""

    tenant_id: str
    user_id: str
    groups: tuple[str, ...] = field(default=())
    role: AuthRole = "user"
    # 冗余字段：便于 agent / tool 层快速 `if state["auth"].anonymous` 判断，
    # 不必每次比对字符串。
    anonymous: bool = False

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"invalid auth role: {self.role!r}")

        if not self.tenant_id:
            raise ValueError("tenant_id must not be empty")
        if not self.user_id:
            raise ValueError("user_id must not be empty")

        # 两字段一致性：role=anonymous ⇔ anonymous=True。
        if (self.role == ROLE_ANONYMOUS) != self.anonymous:
            raise ValueError(
                "auth role and anonymous flag must agree "
                f"(role={self.role!r}, anonymous={self.anonymous})"
            )

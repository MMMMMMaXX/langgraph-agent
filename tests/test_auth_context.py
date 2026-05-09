"""AuthContext / build_auth_context 单测。

覆盖：
- AuthContext 构造规则（role 白名单、空字段拒绝、role/anonymous 一致性、frozen）
- build_auth_context 匿名 fallback 开关：ALLOW_ANONYMOUS_AUTH 开/关、
  各种真值字面量、trim、groups 去空等细节
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.api.schemas import AuthRequest
from app.auth import AnonymousAuthDisabled, AuthContext, build_auth_context
from app.constants.auth import (
    ALLOW_ANONYMOUS_AUTH_ENV,
    ANONYMOUS_TENANT_ID,
    ANONYMOUS_USER_ID,
    ROLE_ADMIN,
    ROLE_ANONYMOUS,
    ROLE_SERVICE,
    ROLE_USER,
)


class TestAuthContext:
    def test_basic_user_context(self) -> None:
        ctx = AuthContext(
            tenant_id="t1",
            user_id="u1",
            groups=("eng", "payments"),
            role=ROLE_USER,
        )
        assert ctx.role == ROLE_USER
        assert ctx.anonymous is False
        assert ctx.groups == ("eng", "payments")

    def test_anonymous_requires_flag(self) -> None:
        # role=anonymous 但 anonymous=False → 拒绝
        with pytest.raises(ValueError, match="anonymous"):
            AuthContext(
                tenant_id="t1",
                user_id="u1",
                role=ROLE_ANONYMOUS,
                anonymous=False,
            )

    def test_anonymous_flag_without_role_rejected(self) -> None:
        with pytest.raises(ValueError, match="anonymous"):
            AuthContext(
                tenant_id="t1",
                user_id="u1",
                role=ROLE_USER,
                anonymous=True,
            )

    def test_anonymous_pair_accepted(self) -> None:
        ctx = AuthContext(
            tenant_id=ANONYMOUS_TENANT_ID,
            user_id=ANONYMOUS_USER_ID,
            role=ROLE_ANONYMOUS,
            anonymous=True,
        )
        assert ctx.anonymous is True

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid auth role"):
            AuthContext(tenant_id="t1", user_id="u1", role="root")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "tenant_id,user_id",
        [("", "u1"), ("t1", "")],
    )
    def test_empty_identity_rejected(self, tenant_id: str, user_id: str) -> None:
        with pytest.raises(ValueError):
            AuthContext(tenant_id=tenant_id, user_id=user_id, role=ROLE_USER)

    def test_frozen(self) -> None:
        ctx = AuthContext(tenant_id="t1", user_id="u1", role=ROLE_USER)
        with pytest.raises(FrozenInstanceError):
            ctx.role = ROLE_ADMIN  # type: ignore[misc]

    def test_all_valid_roles_accepted(self) -> None:
        for role in (ROLE_USER, ROLE_ADMIN, ROLE_SERVICE):
            ctx = AuthContext(tenant_id="t1", user_id="u1", role=role)
            assert ctx.role == role
            assert ctx.anonymous is False


class TestBuildAuthContextFromRequest:
    def test_builds_from_auth_request(self) -> None:
        req = AuthRequest(
            tenant_id="tenant-a", user_id="user-1", groups=["g1", ""], role=ROLE_ADMIN
        )
        ctx = build_auth_context(req)
        assert ctx.tenant_id == "tenant-a"
        assert ctx.user_id == "user-1"
        # 空字符串 group 被过滤
        assert ctx.groups == ("g1",)
        assert ctx.role == ROLE_ADMIN
        assert ctx.anonymous is False

    def test_trims_fields(self) -> None:
        req = AuthRequest(
            tenant_id="  t1 ", user_id=" u1 ", groups=[], role=f" {ROLE_USER} "
        )
        ctx = build_auth_context(req)
        assert ctx.tenant_id == "t1"
        assert ctx.user_id == "u1"
        assert ctx.role == ROLE_USER

    def test_invalid_role_in_request_rejected(self) -> None:
        req = AuthRequest(tenant_id="t1", user_id="u1", groups=[], role="root")
        with pytest.raises(ValueError, match="invalid auth role"):
            build_auth_context(req)

    def test_explicit_anonymous_role_sets_flag(self) -> None:
        req = AuthRequest(tenant_id="t1", user_id="u1", groups=[], role=ROLE_ANONYMOUS)
        ctx = build_auth_context(req)
        assert ctx.anonymous is True


class TestAnonymousFallback:
    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
    def test_fallback_enabled_returns_anonymous(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(ALLOW_ANONYMOUS_AUTH_ENV, value)
        ctx = build_auth_context(None)
        assert ctx.anonymous is True
        assert ctx.role == ROLE_ANONYMOUS
        assert ctx.tenant_id == ANONYMOUS_TENANT_ID
        assert ctx.user_id == ANONYMOUS_USER_ID
        assert ctx.groups == ()

    @pytest.mark.parametrize("value", ["false", "0", "no", "", "anything"])
    def test_fallback_disabled_raises(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(ALLOW_ANONYMOUS_AUTH_ENV, value)
        with pytest.raises(AnonymousAuthDisabled):
            build_auth_context(None)

    def test_fallback_env_unset_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ALLOW_ANONYMOUS_AUTH_ENV, raising=False)
        with pytest.raises(AnonymousAuthDisabled):
            build_auth_context(None)

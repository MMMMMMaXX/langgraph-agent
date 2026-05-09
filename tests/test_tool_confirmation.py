"""Unit tests for app/tools/confirmation.py."""

from __future__ import annotations

import pytest

from app.constants.tool_safety import (
    CONFIRMATION_SECRET_ENV,
    CONFIRMATION_TOKEN_TTL_SECONDS,
    ERR_TOKEN_EXPIRED,
    ERR_TOKEN_INVALID,
    ERR_TOKEN_MISMATCH,
)
from app.tools.confirmation import (
    ConfirmationSecretMissing,
    ExpiredConfirmationToken,
    InvalidConfirmationToken,
    MismatchedConfirmationToken,
    issue_token,
    verify_token,
)
from app.tools.idempotency import compute_args_hash


@pytest.fixture(autouse=True)
def _set_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIRMATION_SECRET_ENV, "unit-test-secret")


def _issue(now: int = 1000, **overrides):
    defaults = dict(
        idempotency_key="sha256:k1",
        tool_name="ticket_create",
        tenant_id="t1",
        user_id="u1",
        args={"title": "Fix payment"},
        now=now,
    )
    defaults.update(overrides)
    return issue_token(**defaults)


def _verify(token: str, now: int = 1000, **overrides):
    defaults = dict(
        tool_name="ticket_create",
        tenant_id="t1",
        user_id="u1",
        args={"title": "Fix payment"},
        now=now,
    )
    defaults.update(overrides)
    return verify_token(token, **defaults)


# ---------- issue + round-trip ----------


def test_issue_returns_payload_with_expires_at() -> None:
    token, payload = _issue(now=1000)
    assert payload.expires_at == 1000 + CONFIRMATION_TOKEN_TTL_SECONDS
    assert payload.args_hash == compute_args_hash({"title": "Fix payment"})
    assert token.count(".") == 1


def test_issue_and_verify_round_trip() -> None:
    token, payload = _issue()
    verified = _verify(token)
    assert verified == payload


def test_token_is_stable_for_same_inputs() -> None:
    t1, _ = _issue(now=1000)
    t2, _ = _issue(now=1000)
    assert t1 == t2  # deterministic signing


# ---------- signature / structure errors ----------


def test_token_with_wrong_signature_is_invalid() -> None:
    token, _ = _issue()
    payload_b64, _, _ = token.partition(".")
    tampered = f"{payload_b64}.AAAA"
    with pytest.raises(InvalidConfirmationToken):
        _verify(tampered)


def test_token_without_dot_is_invalid() -> None:
    with pytest.raises(InvalidConfirmationToken):
        _verify("no-dot-here")


def test_token_with_empty_parts_is_invalid() -> None:
    with pytest.raises(InvalidConfirmationToken):
        _verify(".abc")


def test_token_with_bad_base64_is_invalid() -> None:
    with pytest.raises(InvalidConfirmationToken):
        _verify("@@@.@@@")


def test_token_missing_field_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    # 直接手工构造 payload 缺字段 → invalid。
    import base64
    import hashlib
    import hmac
    import json

    payload = {"idempotency_key": "x"}
    payload_bytes = json.dumps(payload).encode("utf-8")
    sig = hmac.new(b"unit-test-secret", payload_bytes, hashlib.sha256).digest()
    token = (
        base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
        + "."
        + base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    )
    with pytest.raises(InvalidConfirmationToken):
        _verify(token)


def test_token_error_code_on_invalid() -> None:
    with pytest.raises(InvalidConfirmationToken) as exc_info:
        _verify("no-dot-here")
    assert exc_info.value.code == ERR_TOKEN_INVALID


# ---------- expiry ----------


def test_token_expired() -> None:
    token, payload = _issue(now=1000)
    with pytest.raises(ExpiredConfirmationToken) as exc_info:
        _verify(token, now=payload.expires_at + 1)
    assert exc_info.value.code == ERR_TOKEN_EXPIRED


def test_token_valid_right_before_expiry() -> None:
    token, payload = _issue(now=1000)
    _verify(token, now=payload.expires_at - 1)


def test_token_invalid_exactly_at_expiry() -> None:
    token, payload = _issue(now=1000)
    with pytest.raises(ExpiredConfirmationToken):
        _verify(token, now=payload.expires_at)


# ---------- mismatch (misuse detection) ----------


def test_tool_name_mismatch_rejected() -> None:
    token, _ = _issue()
    with pytest.raises(MismatchedConfirmationToken) as exc_info:
        _verify(token, tool_name="other.tool")
    assert exc_info.value.code == ERR_TOKEN_MISMATCH


def test_tenant_mismatch_rejected() -> None:
    token, _ = _issue()
    with pytest.raises(MismatchedConfirmationToken):
        _verify(token, tenant_id="t2")


def test_user_mismatch_rejected() -> None:
    token, _ = _issue()
    with pytest.raises(MismatchedConfirmationToken):
        _verify(token, user_id="u2")


def test_args_mismatch_rejected() -> None:
    token, _ = _issue(args={"title": "original"})
    with pytest.raises(MismatchedConfirmationToken):
        _verify(token, args={"title": "modified"})


def test_idempotency_key_mismatch_rejected() -> None:
    token, _ = _issue()
    with pytest.raises(MismatchedConfirmationToken):
        _verify(token, expected_idempotency_key="sha256:other")


def test_idempotency_key_optional() -> None:
    # 不传 expected_idempotency_key → 只校验其他字段。
    token, _ = _issue()
    _verify(token, expected_idempotency_key=None)


# ---------- secret missing ----------


def test_issue_fails_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONFIRMATION_SECRET_ENV, raising=False)
    with pytest.raises(ConfirmationSecretMissing):
        _issue()


def test_verify_fails_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    token, _ = _issue()
    monkeypatch.delenv(CONFIRMATION_SECRET_ENV, raising=False)
    with pytest.raises(ConfirmationSecretMissing):
        _verify(token)


def test_verify_fails_with_different_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, _ = _issue()
    monkeypatch.setenv(CONFIRMATION_SECRET_ENV, "rotated-secret")
    with pytest.raises(InvalidConfirmationToken):
        _verify(token)

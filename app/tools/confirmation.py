"""Confirmation Token 签发与校验（self-contained）。

设计目标：服务端**不保存** pending state，仅靠 token 自身就能判断二次请求
合法性。跨进程 / 重启场景都稳定。

Token 格式：`base64url(payload_json).base64url(signature_bytes)`

Payload 字段（对应设计文档 §5）：
- `idempotency_key`：绑定到具体业务意图的 hash（sha256:…）
- `tool_name`：防止 A 工具的 token 被用到 B 工具
- `args`：LLM 在 step-1 选定的原始参数，用于 step-2 确定性重放（绕开 LLM）
- `args_hash`：参数的 sha256，防止 token 签发后参数被篡改
- `tenant_id` / `user_id`：防止身份挪用
- `expires_at`：Unix 时间戳（秒），过期后 token 失效

Signature：`hmac_sha256(secret, payload_json_bytes)`，恒定时间比较。

关键纪律：
- secret 从 env `CONFIRMATION_SECRET` 读；未配置时拒签也拒发，失败 fail-closed。
- payload 同时保留 `args_hash` 和明文 `args`：前者用于 verify 时的防篡改校验；
  后者用于 tool_agent 在二次请求里"确定性重放"（跳过 LLM，避免 args 被改写）。
  HMAC 签名保证两者一致；token 本身随 HTTPS 传输，敏感度与业务请求同级。
- 不做 one-time-use：网络抖动会误杀；防重入靠 `tool_executions` 主键 UNIQUE。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from app.constants.tool_safety import (
    CONFIRMATION_SECRET_ENV,
    CONFIRMATION_TOKEN_TTL_SECONDS,
    ERR_TOKEN_EXPIRED,
    ERR_TOKEN_INVALID,
    ERR_TOKEN_MISMATCH,
)
from app.tools.idempotency import compute_args_hash

# payload 里的 key 字段；集中起来，便于校验和序列化共享。
_FIELD_IDEMPOTENCY_KEY = "idempotency_key"
_FIELD_TOOL_NAME = "tool_name"
_FIELD_ARGS = "args"
_FIELD_ARGS_HASH = "args_hash"
_FIELD_TENANT_ID = "tenant_id"
_FIELD_USER_ID = "user_id"
_FIELD_EXPIRES_AT = "expires_at"

_REQUIRED_FIELDS = (
    _FIELD_IDEMPOTENCY_KEY,
    _FIELD_TOOL_NAME,
    _FIELD_ARGS,
    _FIELD_ARGS_HASH,
    _FIELD_TENANT_ID,
    _FIELD_USER_ID,
    _FIELD_EXPIRES_AT,
)


@dataclass(frozen=True)
class ConfirmationPayload:
    """Token 载荷的结构化视图。

    序列化时字段顺序由 `to_json_bytes` 固定（sorted keys），保证签名稳定。
    """

    idempotency_key: str
    tool_name: str
    args: dict[str, Any]
    args_hash: str
    tenant_id: str
    user_id: str
    expires_at: int

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")


class ConfirmationSecretMissing(Exception):
    """`CONFIRMATION_SECRET` 环境变量未配置；任何签发/校验都禁止继续。"""


class ConfirmationTokenError(Exception):
    """Token 校验失败的基类；子类带具体 `code`。"""

    code: str = ERR_TOKEN_INVALID

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.code)


class InvalidConfirmationToken(ConfirmationTokenError):
    code = ERR_TOKEN_INVALID


class ExpiredConfirmationToken(ConfirmationTokenError):
    code = ERR_TOKEN_EXPIRED


class MismatchedConfirmationToken(ConfirmationTokenError):
    """payload 的 tenant / user / tool / args 与当次请求不一致。"""

    code = ERR_TOKEN_MISMATCH


# ---------- helpers ----------


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(encoded: str) -> bytes:
    padding = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(encoded + padding)


def _load_secret() -> bytes:
    secret = os.getenv(CONFIRMATION_SECRET_ENV, "").strip()
    if not secret:
        raise ConfirmationSecretMissing(
            f"{CONFIRMATION_SECRET_ENV} is not configured; "
            "side-effect tool confirmation is disabled."
        )
    return secret.encode("utf-8")


def _sign(payload_bytes: bytes) -> bytes:
    return hmac.new(_load_secret(), payload_bytes, hashlib.sha256).digest()


# ---------- public API ----------


def issue_token(
    *,
    idempotency_key: str,
    tool_name: str,
    tenant_id: str,
    user_id: str,
    args: dict[str, Any],
    now: int | None = None,
    ttl_seconds: int = CONFIRMATION_TOKEN_TTL_SECONDS,
) -> tuple[str, ConfirmationPayload]:
    """签发 token；返回 `(token_str, payload)`。

    `args` 立即求 hash 写进 payload，原始 args 不会进入 token。
    """

    timestamp = now if now is not None else int(time.time())
    payload = ConfirmationPayload(
        idempotency_key=idempotency_key,
        tool_name=tool_name,
        args=dict(args),
        args_hash=compute_args_hash(args),
        tenant_id=tenant_id,
        user_id=user_id,
        expires_at=timestamp + ttl_seconds,
    )
    payload_bytes = payload.to_json_bytes()
    signature = _sign(payload_bytes)
    token = f"{_b64url_encode(payload_bytes)}.{_b64url_encode(signature)}"
    return token, payload


def _parse_token(token: str) -> tuple[ConfirmationPayload, bytes, bytes]:
    """拆分 token → (payload, payload_bytes, signature)。格式异常一律 Invalid。"""

    if not isinstance(token, str) or "." not in token:
        raise InvalidConfirmationToken("malformed token")

    payload_b64, _, signature_b64 = token.partition(".")
    if not payload_b64 or not signature_b64:
        raise InvalidConfirmationToken("malformed token parts")

    try:
        payload_bytes = _b64url_decode(payload_b64)
        signature = _b64url_decode(signature_b64)
        raw = json.loads(payload_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise InvalidConfirmationToken("cannot decode token") from exc

    if not isinstance(raw, dict):
        raise InvalidConfirmationToken("payload is not a json object")
    for field in _REQUIRED_FIELDS:
        if field not in raw:
            raise InvalidConfirmationToken(f"missing field: {field}")

    try:
        raw_args = raw[_FIELD_ARGS]
        if not isinstance(raw_args, dict):
            raise InvalidConfirmationToken("args is not a json object")
        payload = ConfirmationPayload(
            idempotency_key=str(raw[_FIELD_IDEMPOTENCY_KEY]),
            tool_name=str(raw[_FIELD_TOOL_NAME]),
            args=dict(raw_args),
            args_hash=str(raw[_FIELD_ARGS_HASH]),
            tenant_id=str(raw[_FIELD_TENANT_ID]),
            user_id=str(raw[_FIELD_USER_ID]),
            expires_at=int(raw[_FIELD_EXPIRES_AT]),
        )
    except (TypeError, ValueError) as exc:
        raise InvalidConfirmationToken("invalid payload fields") from exc

    return payload, payload_bytes, signature


def verify_token(
    token: str,
    *,
    tool_name: str,
    tenant_id: str,
    user_id: str,
    args: dict[str, Any],
    expected_idempotency_key: str | None = None,
    now: int | None = None,
) -> ConfirmationPayload:
    """校验 token；返回 payload 或抛出子类化的 `ConfirmationTokenError`。

    校验顺序（按"成本从低到高"）：
    1. 结构 / 解码；
    2. 签名（HMAC 恒定时间比较）；
    3. 过期；
    4. 身份 / 工具 / args / idempotency_key 的一致性（防挪用）。

    注意：签名错误不区分"是签错了还是被篡改了"，统一报 invalid。
    """

    payload, payload_bytes, signature = _parse_token(token)

    expected_signature = _sign(payload_bytes)
    if not hmac.compare_digest(expected_signature, signature):
        raise InvalidConfirmationToken("signature mismatch")

    timestamp = now if now is not None else int(time.time())
    if timestamp >= payload.expires_at:
        raise ExpiredConfirmationToken(
            f"token expired at {payload.expires_at}, now={timestamp}"
        )

    # 业务一致性：任何字段不对得上，都按"挪用"处理。
    if payload.tool_name != tool_name:
        raise MismatchedConfirmationToken("tool_name mismatch")
    if payload.tenant_id != tenant_id:
        raise MismatchedConfirmationToken("tenant_id mismatch")
    if payload.user_id != user_id:
        raise MismatchedConfirmationToken("user_id mismatch")

    actual_args_hash = compute_args_hash(args)
    if not hmac.compare_digest(payload.args_hash, actual_args_hash):
        raise MismatchedConfirmationToken("args_hash mismatch")

    if (
        expected_idempotency_key is not None
        and payload.idempotency_key != expected_idempotency_key
    ):
        raise MismatchedConfirmationToken("idempotency_key mismatch")

    return payload


def decode_signed_payload(
    token: str,
    *,
    now: int | None = None,
) -> ConfirmationPayload:
    """解 token 并校验"签名 + 过期"，但不做 tenant / user / tool / args 绑定检查。

    用途：tool_agent 在二次请求里需要先从 token 里拿 `tool_name` 和 `args`，
    才能确定性重放工具调用；这一层仅保证"token 是我签的而且没过期"，业务
    一致性检查留给下游 `verify_token`（在 pipeline 里使用 payload.args 调用，
    args_hash 必定匹配，失败只会来自身份或 tool_name 被挪用）。
    """

    payload, payload_bytes, signature = _parse_token(token)

    expected_signature = _sign(payload_bytes)
    if not hmac.compare_digest(expected_signature, signature):
        raise InvalidConfirmationToken("signature mismatch")

    timestamp = now if now is not None else int(time.time())
    if timestamp >= payload.expires_at:
        raise ExpiredConfirmationToken(
            f"token expired at {payload.expires_at}, now={timestamp}"
        )

    return payload


def redact_pending_confirmation(pending: dict[str, Any] | None) -> dict[str, Any]:
    """去掉 token 本体，保留前端/trace 用得上的元数据 + `token_present` 布尔。

    使用约束：**只有 API 顶层 `pending_confirmation` 才允许携带原始 token**；
    任何会被写入 debug_info / LangSmith / agent_outputs 的 pending_confirmation
    都必须先走这个函数，避免 token 随 trace 泄漏。

    空输入返回空 dict，保持调用点不用判空。
    """

    if not pending:
        return {}
    redacted: dict[str, Any] = {
        "tool_name": pending.get("tool_name", ""),
        "expires_at": pending.get("expires_at", ""),
        "idempotency_key": pending.get("idempotency_key", ""),
        "token_present": bool(pending.get("token")),
    }
    # args 对 Verifier / 展示都有用（tool_name + args 构成"将要执行什么"）；
    # token 本体是唯一必须剔除的字段。
    if "args" in pending:
        redacted["args"] = pending["args"]
    return redacted


__all__ = [
    "ConfirmationPayload",
    "ConfirmationSecretMissing",
    "ConfirmationTokenError",
    "ExpiredConfirmationToken",
    "InvalidConfirmationToken",
    "MismatchedConfirmationToken",
    "decode_signed_payload",
    "issue_token",
    "redact_pending_confirmation",
    "verify_token",
]

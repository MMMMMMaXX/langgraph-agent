"""side_effect 管线（pipeline）与 ticket_create 工具的集成测试。

这里既覆盖 `wrap_side_effect_tool` 在每个分支（匿名 / need_confirmation /
有效 token 首次 WINNER / 有效 token 重放 EXISTING / 签名失败）的行为，
也顺带验证 `ticket_create` 的 UNIQUE 冲突能被 pipeline 兜住。

测试隔离策略：
- 用 monkeypatch 把 execution_record / ticket 模块里的 `OPERATIONS_CONFIG`
  指向 `tmp_path / operations.sqlite3`，避免污染真实 data 目录，也避免并发跑
  时多个测试互相看到对方的 tool_executions / mock_tickets。
- `CONFIRMATION_SECRET` 用 autouse fixture 注入，确保签发/校验都走同一个密钥。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.auth.context import AuthContext
from app.constants.auth import (
    ANONYMOUS_TENANT_ID,
    ANONYMOUS_USER_ID,
    ROLE_ANONYMOUS,
    ROLE_USER,
)
from app.constants.tool_safety import (
    CONFIRMATION_SECRET_ENV,
    ERR_TOKEN_INVALID,
    TOOL_STATUS_FAILED,
    TOOL_STATUS_SUCCEEDED,
)
from app.constants.tooling import TOOL_NAME_TICKET_CREATE
from app.tools import execution_record as exec_record_mod
from app.tools import ticket as ticket_mod
from app.tools.confirmation import issue_token
from app.tools.idempotency import compute_idempotency_key
from app.tools.metadata import get_tool_metadata
from app.tools.pipeline import SideEffectContext, wrap_side_effect_tool
from app.tools.ticket import create_ticket, ticket_create_tool


# ---------------------------- fixtures ----------------------------


@dataclass(frozen=True)
class _TmpOpsConfig:
    path: str


@pytest.fixture(autouse=True)
def _secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIRMATION_SECRET_ENV, "pipeline-test-secret")


@pytest.fixture()
def ops_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """把 tool_executions / mock_tickets 的存储指向临时目录。"""

    db = tmp_path / "operations.sqlite3"
    cfg = _TmpOpsConfig(path=str(db))
    monkeypatch.setattr(exec_record_mod, "OPERATIONS_CONFIG", cfg)
    monkeypatch.setattr(ticket_mod, "OPERATIONS_CONFIG", cfg)
    return db


def _auth(*, anonymous: bool = False, tenant: str = "t1", user: str = "u1") -> AuthContext:
    if anonymous:
        return AuthContext(
            tenant_id=ANONYMOUS_TENANT_ID,
            user_id=ANONYMOUS_USER_ID,
            groups=(),
            role=ROLE_ANONYMOUS,
            anonymous=True,
        )
    return AuthContext(
        tenant_id=tenant,
        user_id=user,
        groups=(),
        role=ROLE_USER,
        anonymous=False,
    )


def _wrap(ctx: SideEffectContext, tool_impl=None):
    meta = get_tool_metadata(TOOL_NAME_TICKET_CREATE)
    return wrap_side_effect_tool(
        TOOL_NAME_TICKET_CREATE,
        tool_impl or ticket_create_tool,
        meta,
        ctx,
    )


# ---------------------------- ticket_create 原子语义 ----------------------------


def test_create_ticket_returns_created_true_on_first_write(ops_db: Path) -> None:
    result = create_ticket(
        title="Reset password",
        idempotency_key="sha256:key-A",
        tenant_id="t1",
        user_id="u1",
    )
    assert result["created"] is True
    assert result["ticket_id"].startswith("TK-")
    assert result["idempotency_key"] == "sha256:key-A"


def test_create_ticket_is_idempotent_on_same_key(ops_db: Path) -> None:
    # 第二次同 key 写入命中 UNIQUE，pipeline 里这是正常的"重放成功"分支。
    first = create_ticket(
        title="X",
        idempotency_key="sha256:key-B",
        tenant_id="t1",
        user_id="u1",
    )
    second = create_ticket(
        title="X2",  # 故意换 title，不影响 idempotency 行为
        idempotency_key="sha256:key-B",
        tenant_id="t1",
        user_id="u1",
    )
    assert second["created"] is False
    assert second["ticket_id"] == first["ticket_id"]


def test_ticket_create_tool_requires_pipeline_injection() -> None:
    # LLM 绕过 pipeline 直接调（理论不会发生，但做 fail-fast 保险）。
    with pytest.raises(ValueError):
        ticket_create_tool(title="bare call")


# ---------------------------- pipeline: anonymous ----------------------------


def test_pipeline_rejects_anonymous_with_audit_record(ops_db: Path) -> None:
    ctx = SideEffectContext(auth=_auth(anonymous=True), session_id="s", request_id="r")
    tool = _wrap(ctx)

    out = tool(title="hi")

    assert "匿名" in out
    # 匿名拒绝记一笔非持久 audit，不会写 tool_executions DB。
    assert ctx.executions == [
        {
            "tool_name": TOOL_NAME_TICKET_CREATE,
            "status": "rejected_anonymous",
            "error": "anonymous_forbidden_side_effect",
        }
    ]
    assert ctx.pending_confirmation is None


# ---------------------------- pipeline: need confirmation ----------------------------


def test_pipeline_without_token_emits_need_confirmation(ops_db: Path) -> None:
    ctx = SideEffectContext(
        auth=_auth(),
        session_id="s1",
        request_id="r1",
        confirmation_token="",  # 首次请求
        now=1000,
    )
    tool = _wrap(ctx)

    out = tool(title="Refund request")

    assert "确认" in out
    assert ctx.pending_confirmation is not None
    pc = ctx.pending_confirmation
    assert pc["tool_name"] == TOOL_NAME_TICKET_CREATE
    assert pc["args"] == {"title": "Refund request"}
    assert pc["token"]  # 真的签发了一张
    # 首次请求不会触碰 tool_executions 表。
    assert ctx.executions == []


# ---------------------------- pipeline: valid token → execute ----------------------------


def test_pipeline_with_valid_token_executes_and_records_success(ops_db: Path) -> None:
    auth = _auth()
    args = {"title": "Fix login"}
    ikey = compute_idempotency_key(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        session_id="s1",
        tool_name=TOOL_NAME_TICKET_CREATE,
        args=args,
    )
    token, _ = issue_token(
        idempotency_key=ikey,
        tool_name=TOOL_NAME_TICKET_CREATE,
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        args=args,
        now=1000,
    )
    ctx = SideEffectContext(
        auth=auth,
        session_id="s1",
        request_id="r1",
        confirmation_token=token,
        now=1000,
    )
    tool = _wrap(ctx)

    out = tool(**args)

    # 返回字符串里含 ticket_id / created=True。
    assert "TK-" in out
    assert "True" in out
    # 一笔 succeeded 的 record。
    assert len(ctx.executions) == 1
    assert ctx.executions[0]["status"] == TOOL_STATUS_SUCCEEDED
    assert ctx.pending_confirmation is None


# ---------------------------- pipeline: idempotent replay ----------------------------


def test_pipeline_replay_with_same_args_returns_cached_record(ops_db: Path) -> None:
    auth = _auth()
    args = {"title": "Dedup me"}
    ikey = compute_idempotency_key(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        session_id="s1",
        tool_name=TOOL_NAME_TICKET_CREATE,
        args=args,
    )
    token, _ = issue_token(
        idempotency_key=ikey,
        tool_name=TOOL_NAME_TICKET_CREATE,
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        args=args,
        now=1000,
    )

    def _new_ctx() -> SideEffectContext:
        return SideEffectContext(
            auth=auth, session_id="s1", request_id="r", confirmation_token=token, now=1000
        )

    ctx1 = _new_ctx()
    out1 = _wrap(ctx1)(**args)
    ctx2 = _new_ctx()
    out2 = _wrap(ctx2)(**args)

    # 第二次命中 EXISTING 分支，返回的是原 record 的 result。
    assert "TK-" in out1 and "TK-" in out2
    assert ctx1.executions[0]["status"] == TOOL_STATUS_SUCCEEDED
    assert ctx2.executions[0]["status"] == TOOL_STATUS_SUCCEEDED
    # 两次返回的 ticket_id 一致（幂等）。
    assert ctx1.executions[0]["idempotency_key"] == ctx2.executions[0]["idempotency_key"]


# ---------------------------- pipeline: invalid token ----------------------------


def test_pipeline_rejects_tampered_token(ops_db: Path) -> None:
    ctx = SideEffectContext(
        auth=_auth(),
        session_id="s1",
        request_id="r1",
        confirmation_token="abc.def",  # 垃圾 token
        now=1000,
    )
    tool = _wrap(ctx)

    out = tool(title="Anything")

    assert ERR_TOKEN_INVALID in out
    # 校验失败不落 execution record（尚未抢占）。
    assert ctx.executions == []


# ---------------------------- pipeline: tool internal failure ----------------------------


def test_pipeline_records_failure_when_tool_raises(ops_db: Path) -> None:
    def boom(**_: Any) -> dict[str, Any]:
        raise RuntimeError("downstream down")

    auth = _auth()
    args = {"title": "Will fail"}
    ikey = compute_idempotency_key(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        session_id="s1",
        tool_name=TOOL_NAME_TICKET_CREATE,
        args=args,
    )
    token, _ = issue_token(
        idempotency_key=ikey,
        tool_name=TOOL_NAME_TICKET_CREATE,
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        args=args,
        now=1000,
    )
    ctx = SideEffectContext(
        auth=auth, session_id="s1", request_id="r", confirmation_token=token, now=1000
    )

    out = _wrap(ctx, tool_impl=boom)(**args)

    assert "失败" in out
    assert len(ctx.executions) == 1
    assert ctx.executions[0]["status"] == TOOL_STATUS_FAILED
    assert "downstream down" in (ctx.executions[0]["error"] or "")

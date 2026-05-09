"""Unit tests for app/tools/execution_record.py."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.constants.tool_safety import (
    TOOL_STATUS_FAILED,
    TOOL_STATUS_PENDING,
    TOOL_STATUS_SUCCEEDED,
    TOOL_STATUS_TIMEOUT_UNKNOWN,
)
from app.tools.execution_record import (
    ExecutionRecordAlreadyExists,
    get_by_key,
    insert_pending,
    update_status,
)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "operations.sqlite3"


def _insert(db_path: Path, key: str = "sha256:k1", **overrides):
    defaults = dict(
        idempotency_key=key,
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        request_id="r1",
        tool_name="ticket_create",
        args_json='{"title":"x"}',
        path=db_path,
        now=1000,
    )
    defaults.update(overrides)
    return insert_pending(**defaults)


# ---------- insert_pending ----------


def test_insert_pending_creates_record(db_path: Path) -> None:
    record = _insert(db_path)
    assert record.status == TOOL_STATUS_PENDING
    assert record.created_at == 1000
    assert record.updated_at == 1000
    assert record.result_json is None
    assert record.error is None


def test_insert_pending_persists_to_db(db_path: Path) -> None:
    _insert(db_path)
    fetched = get_by_key("sha256:k1", path=db_path)
    assert fetched is not None
    assert fetched.tenant_id == "t1"
    assert fetched.status == TOOL_STATUS_PENDING


def test_insert_pending_conflict_raises(db_path: Path) -> None:
    _insert(db_path)
    with pytest.raises(ExecutionRecordAlreadyExists) as exc_info:
        _insert(db_path)
    assert exc_info.value.idempotency_key == "sha256:k1"


def test_insert_pending_empty_key_rejected(db_path: Path) -> None:
    with pytest.raises(ValueError):
        _insert(db_path, key="")


def test_insert_pending_creates_parent_dir(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "dir" / "ops.sqlite3"
    _insert(nested)
    assert nested.exists()


# ---------- get_by_key ----------


def test_get_by_key_returns_none_when_missing(db_path: Path) -> None:
    assert get_by_key("nope", path=db_path) is None


def test_get_by_key_round_trip_args_json(db_path: Path) -> None:
    _insert(db_path, args_json='{"city":"北京","q":1}')
    record = get_by_key("sha256:k1", path=db_path)
    assert record is not None
    assert record.args == {"city": "北京", "q": 1}


# ---------- update_status ----------


def test_update_status_to_succeeded(db_path: Path) -> None:
    _insert(db_path)
    updated = update_status(
        idempotency_key="sha256:k1",
        status=TOOL_STATUS_SUCCEEDED,
        result_json='{"ticket_id":"T-1"}',
        path=db_path,
        now=2000,
    )
    assert updated.status == TOOL_STATUS_SUCCEEDED
    assert updated.result == {"ticket_id": "T-1"}
    assert updated.updated_at == 2000
    assert updated.created_at == 1000  # unchanged


def test_update_status_to_failed_with_error(db_path: Path) -> None:
    _insert(db_path)
    updated = update_status(
        idempotency_key="sha256:k1",
        status=TOOL_STATUS_FAILED,
        error="boom",
        path=db_path,
        now=2000,
    )
    assert updated.status == TOOL_STATUS_FAILED
    assert updated.error == "boom"
    assert updated.result_json is None


def test_update_status_to_timeout_unknown(db_path: Path) -> None:
    _insert(db_path)
    updated = update_status(
        idempotency_key="sha256:k1",
        status=TOOL_STATUS_TIMEOUT_UNKNOWN,
        error="local timeout",
        path=db_path,
    )
    assert updated.status == TOOL_STATUS_TIMEOUT_UNKNOWN


def test_update_status_rejects_pending(db_path: Path) -> None:
    _insert(db_path)
    with pytest.raises(ValueError, match="pending"):
        update_status(
            idempotency_key="sha256:k1",
            status=TOOL_STATUS_PENDING,
            path=db_path,
        )


def test_update_status_rejects_invalid_status(db_path: Path) -> None:
    _insert(db_path)
    with pytest.raises(ValueError, match="invalid status"):
        update_status(
            idempotency_key="sha256:k1",
            status="weird",
            path=db_path,
        )


def test_update_status_unknown_key_raises(db_path: Path) -> None:
    with pytest.raises(KeyError):
        update_status(
            idempotency_key="never",
            status=TOOL_STATUS_SUCCEEDED,
            path=db_path,
        )


# ---------- schema ----------


def test_schema_has_indexes(db_path: Path) -> None:
    _insert(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND tbl_name='tool_executions'"
        ).fetchall()
    names = {row[0] for row in rows}
    assert "idx_tool_exec_session" in names
    assert "idx_tool_exec_user" in names

"""Unit tests for app/tools/idempotency.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.constants.tool_safety import (
    TOOL_STATUS_FAILED,
    TOOL_STATUS_PENDING,
    TOOL_STATUS_SUCCEEDED,
    TOOL_STATUS_TIMEOUT_UNKNOWN,
)
from app.tools.execution_record import insert_pending, update_status
from app.tools.idempotency import (
    AcquireOutcome,
    acquire_or_resolve,
    compute_args_hash,
    compute_idempotency_key,
    finalize_failure,
    finalize_success,
    finalize_timeout_unknown,
)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "operations.sqlite3"


def _acquire(db_path: Path, **overrides):
    defaults = dict(
        idempotency_key="sha256:k1",
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        request_id="r1",
        tool_name="ticket_create",
        args={"title": "x"},
        path=db_path,
    )
    defaults.update(overrides)
    return acquire_or_resolve(**defaults)


# ---------- compute_idempotency_key ----------


def test_idempotency_key_is_stable_across_arg_order() -> None:
    k1 = compute_idempotency_key(
        tenant_id="t",
        user_id="u",
        session_id="s",
        tool_name="ticket_create",
        args={"a": 1, "b": 2},
    )
    k2 = compute_idempotency_key(
        tenant_id="t",
        user_id="u",
        session_id="s",
        tool_name="ticket_create",
        args={"b": 2, "a": 1},
    )
    assert k1 == k2
    assert k1.startswith("sha256:")


def test_idempotency_key_differs_on_any_dimension() -> None:
    base = dict(
        tenant_id="t",
        user_id="u",
        session_id="s",
        tool_name="ticket_create",
        args={"a": 1},
    )
    base_key = compute_idempotency_key(**base)
    for field, new in [
        ("tenant_id", "t2"),
        ("user_id", "u2"),
        ("session_id", "s2"),
        ("tool_name", "other"),
    ]:
        assert compute_idempotency_key(**{**base, field: new}) != base_key

    diff_args = compute_idempotency_key(**{**base, "args": {"a": 2}})
    assert diff_args != base_key


def test_idempotency_key_rejects_empty_dimensions() -> None:
    with pytest.raises(ValueError):
        compute_idempotency_key(
            tenant_id="",
            user_id="u",
            session_id="s",
            tool_name="t",
            args={},
        )


def test_compute_args_hash_is_order_stable() -> None:
    assert compute_args_hash({"a": 1, "b": 2}) == compute_args_hash({"b": 2, "a": 1})


# ---------- acquire_or_resolve ----------


def test_acquire_first_call_is_winner(db_path: Path) -> None:
    result = _acquire(db_path)
    assert result.outcome == AcquireOutcome.WINNER
    assert result.record.status == TOOL_STATUS_PENDING


def test_acquire_second_call_sees_existing_after_success(db_path: Path) -> None:
    first = _acquire(db_path)
    finalize_success(first.record.idempotency_key, {"id": "T-1"}, path=db_path)

    second = _acquire(db_path)
    assert second.outcome == AcquireOutcome.EXISTING
    assert second.record.status == TOOL_STATUS_SUCCEEDED
    assert second.record.result == {"id": "T-1"}


def test_acquire_second_call_sees_existing_after_failure(db_path: Path) -> None:
    first = _acquire(db_path)
    finalize_failure(first.record.idempotency_key, "boom", path=db_path)

    second = _acquire(db_path)
    assert second.outcome == AcquireOutcome.EXISTING
    assert second.record.status == TOOL_STATUS_FAILED
    assert second.record.error == "boom"


def test_acquire_second_call_sees_timeout_unknown(db_path: Path) -> None:
    first = _acquire(db_path)
    finalize_timeout_unknown(first.record.idempotency_key, "t/o", path=db_path)

    second = _acquire(db_path)
    assert second.outcome == AcquireOutcome.EXISTING
    assert second.record.status == TOOL_STATUS_TIMEOUT_UNKNOWN


def test_acquire_polls_until_terminal(db_path: Path) -> None:
    # 先插入 pending；模拟另一进程；在第 2 次 sleep 时转 succeeded。
    first = _acquire(db_path)
    assert first.outcome == AcquireOutcome.WINNER

    sleep_calls: list[float] = []

    def fake_sleep(interval: float) -> None:
        sleep_calls.append(interval)
        if len(sleep_calls) == 2:
            update_status(
                idempotency_key=first.record.idempotency_key,
                status=TOOL_STATUS_SUCCEEDED,
                result_json='{"ok":true}',
                path=db_path,
            )

    second = _acquire(db_path, sleep=fake_sleep)
    assert second.outcome == AcquireOutcome.EXISTING
    assert second.record.status == TOOL_STATUS_SUCCEEDED
    # 至少两次 sleep 才观察到终态。
    assert len(sleep_calls) >= 2


def test_acquire_returns_still_pending_after_max_attempts(db_path: Path) -> None:
    _acquire(db_path)  # winner, leaves status=pending

    sleeps: list[float] = []

    second = _acquire(
        db_path,
        sleep=lambda s: sleeps.append(s),
        poll_interval_ms=10,
        poll_max_attempts=3,
    )
    assert second.outcome == AcquireOutcome.STILL_PENDING
    assert second.record.status == TOOL_STATUS_PENDING
    assert len(sleeps) == 3  # 正好 poll_max_attempts 次


def test_acquire_vanished_record_raises(db_path: Path, monkeypatch) -> None:
    # 先占位制造主键冲突路径，然后把 get_by_key mock 成 None 模拟消失。
    _acquire(db_path)

    import app.tools.idempotency as idem

    monkeypatch.setattr(idem, "get_by_key", lambda *_a, **_kw: None)
    with pytest.raises(RuntimeError, match="vanished"):
        _acquire(db_path, sleep=lambda _: None)


# ---------- finalize_* ----------


def test_finalize_success_writes_result(db_path: Path) -> None:
    insert_pending(
        idempotency_key="sha256:z",
        tenant_id="t",
        user_id="u",
        session_id="s",
        request_id="r",
        tool_name="ticket_create",
        args_json="{}",
        path=db_path,
    )
    record = finalize_success("sha256:z", {"ticket_id": "T-9"}, path=db_path)
    assert record.status == TOOL_STATUS_SUCCEEDED
    assert record.result == {"ticket_id": "T-9"}
    assert record.error is None


def test_finalize_failure_clears_result_json(db_path: Path) -> None:
    insert_pending(
        idempotency_key="sha256:z",
        tenant_id="t",
        user_id="u",
        session_id="s",
        request_id="r",
        tool_name="ticket_create",
        args_json="{}",
        path=db_path,
    )
    record = finalize_failure("sha256:z", "downstream 500", path=db_path)
    assert record.status == TOOL_STATUS_FAILED
    assert record.result_json is None
    assert record.error == "downstream 500"


def test_finalize_timeout_unknown_semantics(db_path: Path) -> None:
    insert_pending(
        idempotency_key="sha256:z",
        tenant_id="t",
        user_id="u",
        session_id="s",
        request_id="r",
        tool_name="ticket_create",
        args_json="{}",
        path=db_path,
    )
    record = finalize_timeout_unknown("sha256:z", "asyncio timeout", path=db_path)
    # 关键：不是 failed，上层禁止把 timeout_unknown 当 failed 处理。
    assert record.status == TOOL_STATUS_TIMEOUT_UNKNOWN
    assert record.status != TOOL_STATUS_FAILED

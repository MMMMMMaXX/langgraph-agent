"""Unit tests for app/tools/monitor.py."""

from __future__ import annotations

from app.constants.tooling import (
    MONITOR_QUERY_ERRORS_BY_SERVICE,
    MONITOR_QUERY_ERRORS_DEFAULT,
    MONITOR_SERVICE_MAX_CHARS,
    TOOL_NAME_MONITOR_QUERY_ERRORS,
)
from app.tools.metadata import TOOL_METADATA
from app.tools.monitor import monitor_query_errors


def test_monitor_returns_service_summary_for_known_service() -> None:
    out = monitor_query_errors("payment-service")
    assert out == MONITOR_QUERY_ERRORS_BY_SERVICE["payment-service"]
    assert "payment-service" in out


def test_monitor_returns_default_for_unknown_service() -> None:
    assert monitor_query_errors("unknown-service") == MONITOR_QUERY_ERRORS_DEFAULT


def test_monitor_returns_default_for_empty_input() -> None:
    assert monitor_query_errors("") == MONITOR_QUERY_ERRORS_DEFAULT
    assert monitor_query_errors("   ") == MONITOR_QUERY_ERRORS_DEFAULT


def test_monitor_rejects_overlong_service_name() -> None:
    too_long = "x" * (MONITOR_SERVICE_MAX_CHARS + 1)
    assert monitor_query_errors(too_long) == MONITOR_QUERY_ERRORS_DEFAULT


def test_monitor_registered_as_read_only_in_metadata() -> None:
    # 防御：把 monitor 从 side_effect 错配会让 Verifier 额外上风险提示；
    # 这里守住 metadata 一致性，避免日后误改。
    meta = TOOL_METADATA[TOOL_NAME_MONITOR_QUERY_ERRORS]
    assert meta.read_only is True
    assert meta.side_effect is False
    assert meta.requires_confirmation is False
    assert meta.idempotency_required is False

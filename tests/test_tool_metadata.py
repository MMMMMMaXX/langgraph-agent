"""Unit tests for app/tools/metadata.py."""

from __future__ import annotations

import pytest

from app.constants.tool_safety import (
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    ERR_TOOL_NOT_REGISTERED,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_LOW,
    RISK_LEVEL_MEDIUM,
)
from app.constants.tooling import TOOL_NAME_CALCULATE, TOOL_NAME_GET_WEATHER
from app.tools.metadata import (
    TOOL_METADATA,
    ToolMetadata,
    ToolNotRegisteredError,
    filter_tools_for_auth,
    get_tool_metadata,
)


def _valid_readonly_kwargs(**overrides):
    base = {
        "name": "t",
        "read_only": True,
        "side_effect": False,
        "requires_confirmation": False,
        "idempotency_required": False,
        "risk_level": RISK_LEVEL_LOW,
    }
    base.update(overrides)
    return base


def _valid_side_effect_kwargs(**overrides):
    base = {
        "name": "t",
        "read_only": False,
        "side_effect": True,
        "requires_confirmation": True,
        "idempotency_required": True,
        "risk_level": RISK_LEVEL_MEDIUM,
    }
    base.update(overrides)
    return base


# -------- ToolMetadata invariants --------


def test_metadata_valid_readonly() -> None:
    meta = ToolMetadata(**_valid_readonly_kwargs())
    assert meta.read_only is True
    assert meta.side_effect is False
    assert meta.timeout_seconds == DEFAULT_TOOL_TIMEOUT_SECONDS


def test_metadata_valid_side_effect() -> None:
    meta = ToolMetadata(**_valid_side_effect_kwargs(risk_level=RISK_LEVEL_HIGH))
    assert meta.side_effect is True
    assert meta.requires_confirmation is True
    assert meta.idempotency_required is True


def test_metadata_empty_name_rejected() -> None:
    with pytest.raises(ValueError, match="tool name"):
        ToolMetadata(**_valid_readonly_kwargs(name=""))


def test_metadata_invalid_risk_level_rejected() -> None:
    with pytest.raises(ValueError, match="risk level"):
        ToolMetadata(**_valid_readonly_kwargs(risk_level="critical"))


def test_metadata_readonly_and_side_effect_both_true_rejected() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ToolMetadata(
            **_valid_readonly_kwargs(
                read_only=True,
                side_effect=True,
                requires_confirmation=True,
                idempotency_required=True,
            )
        )


def test_metadata_readonly_and_side_effect_both_false_rejected() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ToolMetadata(**_valid_readonly_kwargs(read_only=False, side_effect=False))


def test_metadata_readonly_requires_confirmation_rejected() -> None:
    with pytest.raises(ValueError, match="read_only"):
        ToolMetadata(**_valid_readonly_kwargs(requires_confirmation=True))


def test_metadata_readonly_idempotency_required_rejected() -> None:
    with pytest.raises(ValueError, match="read_only"):
        ToolMetadata(**_valid_readonly_kwargs(idempotency_required=True))


def test_metadata_side_effect_without_confirmation_rejected() -> None:
    with pytest.raises(ValueError, match="side_effect"):
        ToolMetadata(**_valid_side_effect_kwargs(requires_confirmation=False))


def test_metadata_side_effect_without_idempotency_rejected() -> None:
    with pytest.raises(ValueError, match="side_effect"):
        ToolMetadata(**_valid_side_effect_kwargs(idempotency_required=False))


def test_metadata_non_positive_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        ToolMetadata(**_valid_readonly_kwargs(timeout_seconds=0))


def test_metadata_frozen() -> None:
    meta = ToolMetadata(**_valid_readonly_kwargs())
    with pytest.raises(Exception):
        meta.name = "x"  # type: ignore[misc]


# -------- TOOL_METADATA registry --------


def test_registry_contains_existing_tools() -> None:
    assert TOOL_NAME_GET_WEATHER in TOOL_METADATA
    assert TOOL_NAME_CALCULATE in TOOL_METADATA
    # read_only 工具：两个；side_effect 工具：ticket_create（PR 4 起）。
    assert TOOL_METADATA[TOOL_NAME_GET_WEATHER].read_only is True
    assert TOOL_METADATA[TOOL_NAME_CALCULATE].read_only is True


def test_get_tool_metadata_returns_registry_entry() -> None:
    assert (
        get_tool_metadata(TOOL_NAME_GET_WEATHER) is TOOL_METADATA[TOOL_NAME_GET_WEATHER]
    )


def test_get_tool_metadata_unknown_raises() -> None:
    with pytest.raises(ToolNotRegisteredError) as exc_info:
        get_tool_metadata("ghost_tool")
    assert exc_info.value.tool_name == "ghost_tool"
    assert ERR_TOOL_NOT_REGISTERED in str(exc_info.value)
    # Backward-compatible: ToolNotRegisteredError is a KeyError.
    assert isinstance(exc_info.value, KeyError)


# -------- filter_tools_for_auth --------


def _spec(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def test_filter_readonly_tools_unchanged_for_anonymous() -> None:
    tools = [_spec(TOOL_NAME_GET_WEATHER), _spec(TOOL_NAME_CALCULATE)]
    impls = {
        TOOL_NAME_GET_WEATHER: lambda **_: "w",
        TOOL_NAME_CALCULATE: lambda **_: "c",
    }

    filtered_tools, filtered_impls = filter_tools_for_auth(tools, impls, anonymous=True)

    assert [t["function"]["name"] for t in filtered_tools] == [
        TOOL_NAME_GET_WEATHER,
        TOOL_NAME_CALCULATE,
    ]
    assert set(filtered_impls.keys()) == {TOOL_NAME_GET_WEATHER, TOOL_NAME_CALCULATE}


def test_filter_removes_side_effect_tool_for_anonymous(monkeypatch) -> None:
    side_effect_meta = ToolMetadata(**_valid_side_effect_kwargs(name="ticket_create"))
    monkeypatch.setitem(TOOL_METADATA, "ticket_create", side_effect_meta)

    tools = [_spec(TOOL_NAME_GET_WEATHER), _spec("ticket_create")]
    impls = {TOOL_NAME_GET_WEATHER: lambda **_: "w", "ticket_create": lambda **_: "t"}

    filtered_tools, filtered_impls = filter_tools_for_auth(tools, impls, anonymous=True)

    assert [t["function"]["name"] for t in filtered_tools] == [TOOL_NAME_GET_WEATHER]
    assert "ticket_create" not in filtered_impls


def test_filter_keeps_side_effect_tool_for_non_anonymous(monkeypatch) -> None:
    side_effect_meta = ToolMetadata(**_valid_side_effect_kwargs(name="ticket_create"))
    monkeypatch.setitem(TOOL_METADATA, "ticket_create", side_effect_meta)

    tools = [_spec(TOOL_NAME_GET_WEATHER), _spec("ticket_create")]
    impls = {TOOL_NAME_GET_WEATHER: lambda **_: "w", "ticket_create": lambda **_: "t"}

    filtered_tools, filtered_impls = filter_tools_for_auth(
        tools, impls, anonymous=False
    )

    names = [t["function"]["name"] for t in filtered_tools]
    assert names == [TOOL_NAME_GET_WEATHER, "ticket_create"]
    assert set(filtered_impls.keys()) == {TOOL_NAME_GET_WEATHER, "ticket_create"}


def test_filter_unregistered_tool_raises() -> None:
    tools = [_spec("mystery")]
    with pytest.raises(ToolNotRegisteredError):
        filter_tools_for_auth(tools, {}, anonymous=False)


def test_filter_invalid_spec_raises() -> None:
    with pytest.raises(ValueError, match="invalid tool spec"):
        filter_tools_for_auth([{"type": "function"}], {}, anonymous=False)

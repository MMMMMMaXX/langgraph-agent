"""Tests for Phase 2 workflow metric aggregation in scripts/eval_chat.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# eval_chat 顶层会做一些环境副作用（改 os.environ / 临时文件），
# import 本身不应抛错；用这次 import 兼带防御住未来回归。
from scripts import eval_chat  # noqa: E402


def _workflow_case() -> dict:
    return {"id": "wf", "category": "workflow"}


def test_workflow_metrics_skipped_for_non_workflow_case() -> None:
    out = eval_chat.collect_workflow_metrics(
        {"id": "x", "category": "knowledge"}, {}, {}, []
    )
    assert out == {}


def test_workflow_metrics_success_path() -> None:
    debug = {
        "planner": {"status": "ok"},
        "composer": {"workflow_status": "succeeded"},
    }
    out = eval_chat.collect_workflow_metrics(_workflow_case(), debug, {}, [])
    assert out["plan_schema_pass_rate"] == "true"
    assert out["workflow_success_rate"] == "true"
    # 单步无重放机会 → 占位 "-"，不进分母。
    assert out["confirmation_bridge_rate"] == "-"


def test_workflow_metrics_need_confirmation_counts_as_closed_loop() -> None:
    debug = {
        "planner": {"status": "ok"},
        "composer": {"workflow_status": "need_confirmation"},
    }
    out = eval_chat.collect_workflow_metrics(_workflow_case(), debug, {}, [])
    assert out["workflow_success_rate"] == "true"


def test_workflow_metrics_planner_failed_marks_schema_miss() -> None:
    debug = {
        "planner": {"status": "failed"},
        "composer": {"workflow_status": "failed"},
    }
    out = eval_chat.collect_workflow_metrics(_workflow_case(), debug, {}, [])
    assert out["plan_schema_pass_rate"] == "false"
    assert out["workflow_success_rate"] == "false"


def test_workflow_metrics_confirmation_bridge_succeeds_when_replay_landed() -> None:
    debug = {
        "planner": {"status": "ok"},
        "composer": {"workflow_status": "succeeded"},
    }
    steps = [
        {"pending_confirmation": True, "tool_executions": []},
        {
            "pending_confirmation": False,
            "tool_executions": [{"status": "succeeded"}],
        },
    ]
    out = eval_chat.collect_workflow_metrics(_workflow_case(), debug, {}, steps)
    assert out["confirmation_bridge_rate"] == "true"


def test_workflow_metrics_confirmation_bridge_fails_when_last_still_pending() -> None:
    debug = {
        "planner": {"status": "ok"},
        "composer": {"workflow_status": "need_confirmation"},
    }
    steps = [
        {"pending_confirmation": True, "tool_executions": []},
        {"pending_confirmation": True, "tool_executions": []},
    ]
    out = eval_chat.collect_workflow_metrics(_workflow_case(), debug, {}, steps)
    assert out["confirmation_bridge_rate"] == "false"

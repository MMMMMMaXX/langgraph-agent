"""Unit tests: app/agents/composer_agent.py."""

from __future__ import annotations

from app.agents.composer_agent import composer_node
from app.auth.context import AuthContext
from app.constants.multi_hop import (
    DEGRADE_NOTICE_LABELS,
    DEGRADE_REASON_BUDGET_EXCEEDED,
    MULTI_HOP_STEP_ID,
)
from app.constants.workflow import (
    COMPOSER_FALLBACK_ALL_FAILED,
    COMPOSER_FALLBACK_NEED_CLARIFICATION,
    COMPOSER_FALLBACK_NEED_CONFIRMATION,
    COMPOSER_FALLBACK_PLAN_FAILED,
    COMPOSER_OUTPUT_KEY,
    ERR_PLAN_LLM_FAILED,
    RISK_WARN_HIGH_RISK_TOOL,
    RISK_WARN_LABELS,
    RISK_WARN_SIDE_EFFECT_CONFIRMED,
    STEP_STATUS_FAILED,
    STEP_STATUS_NEED_CONFIRMATION,
    STEP_STATUS_SKIPPED,
    STEP_STATUS_SUCCEEDED,
    TASK_TYPE_MULTI_HOP_RAG,
    WORKFLOW_STATUS_FAILED,
    WORKFLOW_STATUS_NEED_CLARIFICATION,
    WORKFLOW_STATUS_NEED_CONFIRMATION,
    WORKFLOW_STATUS_PARTIAL,
    WORKFLOW_STATUS_SUCCEEDED,
)


def _auth() -> AuthContext:
    return AuthContext(tenant_id="t1", user_id="u1", role="user", anonymous=False)


def _state(**overrides) -> dict:
    base = {
        "messages": [{"role": "user", "content": "composer test"}],
        "auth": _auth(),
        "request_id": "req-c",
        "session_id": "sess-c",
        "plan": {},
        "plan_id": "plan-c",
        "step_results": {},
        "workflow_status": WORKFLOW_STATUS_SUCCEEDED,
        "verification": {
            "status": "pass",
            "missing_fields": [],
            "unsupported_claims": [],
            "risk_warnings": [],
        },
        "pending_confirmation": {},
    }
    base.update(overrides)
    return base


def _plan(steps: list[dict]) -> dict:
    return {"task_type": "t", "steps": steps, "compose_goal": ""}


def _tool_step(step_id: str, tool: str, **extra) -> dict:
    return {
        "id": step_id,
        "agent": "tool_agent",
        "purpose": extra.get("purpose", "p"),
        "tool": tool,
        "args": extra.get("args", {}),
        "query": None,
        "depends_on": [],
    }


# ---------------------------- fail-closed (empty plan) ----------------------------


def test_composer_fails_closed_on_empty_plan() -> None:
    state = _state(
        plan={},
        workflow_status=WORKFLOW_STATUS_FAILED,
        verification={
            "status": "failed",
            "missing_fields": [],
            "unsupported_claims": [ERR_PLAN_LLM_FAILED],
            "risk_warnings": [],
        },
    )
    out = composer_node(state)
    assert out["answer"].startswith(COMPOSER_FALLBACK_PLAN_FAILED)
    assert ERR_PLAN_LLM_FAILED in out["answer"]
    composer = out["agent_outputs"][COMPOSER_OUTPUT_KEY]
    assert composer["completed_actions"] == []
    assert composer["workflow_status"] == WORKFLOW_STATUS_FAILED


# ---------------------------- success happy path ----------------------------


def test_composer_success_joins_step_outputs_in_order() -> None:
    plan = _plan(
        [
            _tool_step("s1", "get_weather"),
            _tool_step("s2", "calculate"),
        ]
    )
    step_results = {
        "s1": {
            "status": STEP_STATUS_SUCCEEDED,
            "output": "北京天气晴。",
            "tool_name": "get_weather",
        },
        "s2": {
            "status": STEP_STATUS_SUCCEEDED,
            "output": "结果是 3",
            "tool_name": "calculate",
        },
    }
    out = composer_node(_state(plan=plan, step_results=step_results))
    # 顺序拼接
    assert out["answer"].index("北京天气晴") < out["answer"].index("结果是 3")
    composer = out["agent_outputs"][COMPOSER_OUTPUT_KEY]
    assert [a["step"] for a in composer["completed_actions"]] == ["s1", "s2"]
    assert composer["pending_confirmations"] == []
    assert composer["missing_information"] == []
    assert composer["risk_warnings"] == []


def test_composer_appends_risk_warnings_as_user_text() -> None:
    plan = _plan([_tool_step("s1", "get_weather")])
    step_results = {
        "s1": {
            "status": STEP_STATUS_SUCCEEDED,
            "output": "出结果",
            "tool_name": "get_weather",
        }
    }
    verification = {
        "status": "pass",
        "missing_fields": [],
        "unsupported_claims": [],
        "risk_warnings": [RISK_WARN_HIGH_RISK_TOOL, RISK_WARN_SIDE_EFFECT_CONFIRMED],
    }
    out = composer_node(
        _state(plan=plan, step_results=step_results, verification=verification)
    )
    answer = out["answer"]
    assert RISK_WARN_LABELS[RISK_WARN_HIGH_RISK_TOOL] in answer
    assert RISK_WARN_LABELS[RISK_WARN_SIDE_EFFECT_CONFIRMED] in answer


# ---------------------------- need_confirmation ----------------------------


def test_composer_need_confirmation_surfaces_pending_token() -> None:
    plan = _plan([_tool_step("s1", "ticket.create", args={"title": "bug"})])
    step_results = {
        "s1": {
            "status": STEP_STATUS_NEED_CONFIRMATION,
            "output": "请确认 ticket.create 后再试，token=abc",
            "pending_confirmation": {"token": "abc", "tool_name": "ticket_create"},
        }
    }
    pending = {
        "tool_name": "ticket_create",
        "token": "abc",
        "expires_at": "2026-05-12T00:00:00Z",
        "idempotency_key": "ik-1",
    }
    out = composer_node(
        _state(
            plan=plan,
            step_results=step_results,
            workflow_status=WORKFLOW_STATUS_NEED_CONFIRMATION,
            pending_confirmation=pending,
        )
    )
    assert "请确认" in out["answer"]
    composer = out["agent_outputs"][COMPOSER_OUTPUT_KEY]
    assert composer["pending_confirmations"] == [
        {
            "tool": "ticket_create",
            "expires_at": "2026-05-12T00:00:00Z",
            "idempotency_key": "ik-1",
            "token_present": True,
        }
    ]


def test_composer_need_confirmation_redacted_pending_still_marks_token_present() -> (
    None
):
    """workflow_executor 进 composer 前，pending_confirmation 已走 `redact_pending_confirmation`：
    只剩 `token_present=True`，没有 `token` 原文。composer 必须识别这种形态，
    否则 agent_outputs 里的 token_present 会误判为 False，debug / 前端视图就拿
    不到"这里需要确认"的结构化信号。
    """

    plan = _plan([_tool_step("s1", "ticket.create", args={"title": "bug"})])
    step_results = {
        "s1": {
            "status": STEP_STATUS_NEED_CONFIRMATION,
            "output": "请确认 ticket.create",
        }
    }
    pending_redacted = {
        "tool_name": "ticket_create",
        "expires_at": "2026-05-12T00:00:00Z",
        "idempotency_key": "ik-1",
        "token_present": True,
        "args": {"title": "bug"},
    }
    out = composer_node(
        _state(
            plan=plan,
            step_results=step_results,
            workflow_status=WORKFLOW_STATUS_NEED_CONFIRMATION,
            pending_confirmation=pending_redacted,
        )
    )
    composer = out["agent_outputs"][COMPOSER_OUTPUT_KEY]
    assert composer["pending_confirmations"] == [
        {
            "tool": "ticket_create",
            "expires_at": "2026-05-12T00:00:00Z",
            "idempotency_key": "ik-1",
            "token_present": True,
        }
    ]


def test_composer_need_confirmation_falls_back_when_step_output_empty() -> None:
    plan = _plan([_tool_step("s1", "ticket.create", args={"title": "bug"})])
    step_results = {"s1": {"status": STEP_STATUS_NEED_CONFIRMATION, "output": ""}}
    out = composer_node(
        _state(
            plan=plan,
            step_results=step_results,
            workflow_status=WORKFLOW_STATUS_NEED_CONFIRMATION,
        )
    )
    assert COMPOSER_FALLBACK_NEED_CONFIRMATION in out["answer"]


# ---------------------------- need_clarification ----------------------------


def test_composer_need_clarification_lists_missing_fields() -> None:
    plan = _plan([_tool_step("s1", "get_weather", args={})])
    step_results = {"s1": {"status": STEP_STATUS_SUCCEEDED, "output": ""}}
    verification = {
        "status": "need_clarification",
        "missing_fields": ["s1.city"],
        "unsupported_claims": [],
        "risk_warnings": [],
    }
    out = composer_node(
        _state(
            plan=plan,
            step_results=step_results,
            workflow_status=WORKFLOW_STATUS_NEED_CLARIFICATION,
            verification=verification,
        )
    )
    assert COMPOSER_FALLBACK_NEED_CLARIFICATION in out["answer"]
    assert "s1.city" in out["answer"]
    composer = out["agent_outputs"][COMPOSER_OUTPUT_KEY]
    assert composer["missing_information"] == ["s1.city"]


# ---------------------------- partial / failed ----------------------------


def test_composer_partial_preserves_success_and_lists_failures() -> None:
    plan = _plan(
        [
            _tool_step("s1", "get_weather"),
            _tool_step("s2", "calculate"),
        ]
    )
    step_results = {
        "s1": {"status": STEP_STATUS_SUCCEEDED, "output": "北京晴。"},
        "s2": {"status": STEP_STATUS_FAILED, "output": "", "error": "boom"},
    }
    out = composer_node(
        _state(
            plan=plan,
            step_results=step_results,
            workflow_status=WORKFLOW_STATUS_PARTIAL,
        )
    )
    assert "北京晴" in out["answer"]
    assert "s2" in out["answer"]


def test_composer_failed_with_skipped_steps_lists_them() -> None:
    plan = _plan(
        [
            _tool_step("s1", "get_weather"),
            _tool_step("s2", "calculate"),
        ]
    )
    step_results = {
        "s1": {"status": STEP_STATUS_FAILED, "output": "", "error": "net"},
        "s2": {"status": STEP_STATUS_SKIPPED, "output": "", "reason": "upstream"},
    }
    out = composer_node(
        _state(
            plan=plan,
            step_results=step_results,
            workflow_status=WORKFLOW_STATUS_FAILED,
        )
    )
    # 无成功片段 → 至少给失败/跳过归因
    assert "s1" in out["answer"] or COMPOSER_FALLBACK_ALL_FAILED in out["answer"]


def test_composer_failed_with_no_step_info_uses_fallback() -> None:
    plan = _plan([_tool_step("s1", "get_weather")])
    step_results: dict = {"s1": {"status": STEP_STATUS_FAILED, "output": ""}}
    out = composer_node(
        _state(
            plan=plan,
            step_results=step_results,
            workflow_status=WORKFLOW_STATUS_FAILED,
        )
    )
    # 只有失败无成功片段时，答案至少包含失败说明
    assert "s1" in out["answer"] or COMPOSER_FALLBACK_ALL_FAILED in out["answer"]


# ---------------------------- streaming hook ----------------------------


def test_composer_emits_answer_via_stream_callback(monkeypatch) -> None:
    captured: list[tuple[str, dict]] = []

    def fake_stream(event: str, payload: dict) -> None:
        captured.append((event, payload))

    # 注入运行时 stream_callback。
    from app import runtime_context

    monkeypatch.setattr(runtime_context, "get_stream_callback", lambda: fake_stream)
    # composer_agent 已在模块顶部 `from app.runtime_context import get_stream_callback`，
    # 所以同步 patch composer_agent 内引用。
    from app.agents import composer_agent as composer_mod

    monkeypatch.setattr(composer_mod, "get_stream_callback", lambda: fake_stream)

    plan = _plan([_tool_step("s1", "get_weather")])
    step_results = {"s1": {"status": STEP_STATUS_SUCCEEDED, "output": "OK"}}
    out = composer_node(_state(plan=plan, step_results=step_results))
    assert out.get("streamed_answer") is True
    chunks = [p for e, p in captured if e == "chunk"]
    assert chunks and chunks[0]["delta"] == out["answer"]


# ---------------------------- multi-hop passthrough (PR-4) ----------------------------


def _mh_plan() -> dict:
    # multi-hop plan 的 steps 只有一个 pseudo-step；Composer 直通路径不遍历它。
    return {
        "task_type": TASK_TYPE_MULTI_HOP_RAG,
        "steps": [{"id": MULTI_HOP_STEP_ID, "agent": "rag_agent", "depends_on": []}],
        "compose_goal": "",
    }


def test_composer_multi_hop_passthrough_preserves_answer_verbatim() -> None:
    """multi-hop 分支直接透传 step_results[mh1].output，不走二次合成。"""

    fixed_answer = "这是 multi-hop 给出的终稿答案[1][2]。"
    step_results = {
        MULTI_HOP_STEP_ID: {
            "status": STEP_STATUS_SUCCEEDED,
            "output": fixed_answer,
            "citations": [],
            "meta": {},
        }
    }
    out = composer_node(_state(plan=_mh_plan(), step_results=step_results))
    assert out["answer"] == fixed_answer
    composer = out["agent_outputs"][COMPOSER_OUTPUT_KEY]
    # 单条 mh 摘要，不展开 subqueries
    assert [a["step"] for a in composer["completed_actions"]] == [MULTI_HOP_STEP_ID]


def test_composer_multi_hop_partial_appends_degrade_notice() -> None:
    """PARTIAL + degrade_reason=budget_exceeded → 末尾追加对应中文提示。"""

    step_results = {
        MULTI_HOP_STEP_ID: {
            "status": STEP_STATUS_SUCCEEDED,
            "output": "部分答案。",
            "citations": [],
            "meta": {"degrade_reason": "budget_exceeded"},
        }
    }
    out = composer_node(
        _state(
            plan=_mh_plan(),
            step_results=step_results,
            workflow_status=WORKFLOW_STATUS_PARTIAL,
        )
    )
    assert DEGRADE_NOTICE_LABELS[DEGRADE_REASON_BUDGET_EXCEEDED] in out["answer"]
    assert out["answer"].startswith("部分答案。")


def test_composer_multi_hop_citations_merge_subquery_ids() -> None:
    """同 (doc_id, ref) 的多 citations 应合并为一条并合并 subquery_ids。"""

    step_results = {
        MULTI_HOP_STEP_ID: {
            "status": STEP_STATUS_SUCCEEDED,
            "output": "ok",
            "citations": [
                {
                    "ref": "1",
                    "doc_id": "d1",
                    "chunk_id": "c1",
                    "subquery_ids": ("sq1",),
                },
                {
                    "ref": "1",
                    "doc_id": "d1",
                    "chunk_id": "c1",
                    "subquery_ids": ("sq3",),
                },
                {
                    "ref": "2",
                    "doc_id": "d2",
                    "chunk_id": "c2",
                    "subquery_ids": ("sq2",),
                },
            ],
            "meta": {},
        }
    }
    out = composer_node(_state(plan=_mh_plan(), step_results=step_results))
    citations = out["agent_outputs"][COMPOSER_OUTPUT_KEY]["citations"]
    assert len(citations) == 2
    merged = next(c for c in citations if c["doc_id"] == "d1")
    assert merged["subquery_ids"] == ("sq1", "sq3")
    assert merged["step"] == MULTI_HOP_STEP_ID


def test_composer_non_multi_hop_task_ignores_mh_branch() -> None:
    """task_type != multi_hop_rag → 正常 _compose_success_answer 路径不受影响。"""

    plan = _plan([_tool_step("s1", "get_weather")])
    step_results = {
        "s1": {"status": STEP_STATUS_SUCCEEDED, "output": "北京晴。"},
        # 哪怕 step_results 里偶然带 mh1，没有 task_type 也不应命中透传
        MULTI_HOP_STEP_ID: {
            "status": STEP_STATUS_SUCCEEDED,
            "output": "不该出现",
            "citations": [],
            "meta": {},
        },
    }
    out = composer_node(_state(plan=plan, step_results=step_results))
    assert "北京晴" in out["answer"]
    assert "不该出现" not in out["answer"]

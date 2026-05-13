"""Composer Agent：workflow 末端的文字合成节点（Phase 2 PR-4）。

职责边界（对齐 `docs/phase2-workflow-agent.md` §7）：

- **只做合成，不再产生副作用**：Composer 禁止调用工具、禁止写 execution_record。
  所有副作用入口都收敛在 tool_agent，Phase 1 的安全闭环不会被绕过。
- **MVP 采用模板而非 LLM**：
  - Planner / Verifier 已经把结果结构化为 `plan / step_results / verification /
    pending_confirmation`，Composer 把它们按 `workflow_status` 分支拼成中文回答。
  - 不走 LLM 能保证确定性（eval 好写、成本低、延迟稳）；之后若业务需要"润色"
    再引入可切换的 LLM 路径，接口已经把可展示字段拆出来了，替换成本低。
- **输出双写**：
  - `state["answer"]`：直接给 `memory_node` / API 层取用。workflow 支路绕开
    原 `merge_node` 的 LLM 合成（graph 里 `verifier → composer → memory`），
    因为 merge_node 面向"多 agent 输出需要自然语言再融合"的场景，对我们已经
    结构化的 step_results 反而会引入不确定性。
  - `state["agent_outputs"]["composer"]`：结构化字段
    `{completed_actions, pending_confirmations, missing_information, citations,
    risk_warnings}`，供前端展示、eval 断言、后续 Composer 升级使用。
- **风险提示文案映射**：`RISK_WARN_LABELS` 集中维护；Composer 不再自拼中文，
  避免"加了风险 code 却忘改文案"这种跨模块副本问题。
"""

from __future__ import annotations

from typing import Any

from app.constants.workflow import (
    COMPOSER_FALLBACK_ALL_FAILED,
    COMPOSER_FALLBACK_NEED_CLARIFICATION,
    COMPOSER_FALLBACK_NEED_CONFIRMATION,
    COMPOSER_FALLBACK_PLAN_FAILED,
    COMPOSER_OUTPUT_KEY,
    NODE_COMPOSER,
    RISK_WARN_LABELS,
    STEP_AGENT_RAG,
    STEP_AGENT_TOOL,
    STEP_STATUS_FAILED,
    STEP_STATUS_NEED_CONFIRMATION,
    STEP_STATUS_SKIPPED,
    STEP_STATUS_SUCCEEDED,
    WORKFLOW_STATUS_FAILED,
    WORKFLOW_STATUS_NEED_CLARIFICATION,
    WORKFLOW_STATUS_NEED_CONFIRMATION,
    WORKFLOW_STATUS_PARTIAL,
    WORKFLOW_STATUS_SUCCEEDED,
)
from app.runtime_context import get_stream_callback
from app.state import AgentState
from app.utils.logger import log_node, preview

# --------------------------------------------------------------------------
# 结构化片段构造
# --------------------------------------------------------------------------


def _build_completed_actions(
    steps: list[dict[str, Any]],
    step_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """把 succeeded 的 step 摘要成 `{step, agent, tool, summary}` 列表。

    `summary` 截断到 200 字，前端列表场景够用；全文保留在 step_results.output
    以便调试/展开。
    """

    actions: list[dict[str, Any]] = []
    for step in steps:
        step_id = step.get("id", "")
        res = step_results.get(step_id) or {}
        if res.get("status") != STEP_STATUS_SUCCEEDED:
            continue
        actions.append(
            {
                "step": step_id,
                "agent": step.get("agent", ""),
                "tool": step.get("tool") or res.get("tool_name") or "",
                "purpose": step.get("purpose", ""),
                "summary": preview(str(res.get("output") or ""), 200),
            }
        )
    return actions


def _build_pending_confirmations(
    pending_confirmation: dict[str, Any],
) -> list[dict[str, Any]]:
    """单步短路模型下最多一个 pending confirmation；用 list 是为了前端结构兼容。

    agent_outputs 会被写入 debug / trace，这里刻意**不**携带 token 本体，只暴露
    `token_present` 布尔。客户端要拿原始 token 请从 API 响应顶层 `pending_confirmation`
    字段读取（由 chat_runner 透出），而不是从 composer 结构体里翻。
    """

    if not pending_confirmation:
        return []
    # workflow 路径进 composer 之前，pending_confirmation 已经被
    # `redact_pending_confirmation` 脱敏过，只剩 `token_present=True` 而没有
    # `token` 原文；tool_agent 单 turn 兜底路径可能仍然带着 `token`（老调用点）。
    # 两种形态都要识别，否则脱敏路径上 token_present 会被误判为 False。
    token_present = bool(
        pending_confirmation.get("token_present") or pending_confirmation.get("token")
    )
    return [
        {
            "tool": pending_confirmation.get("tool_name", ""),
            "expires_at": pending_confirmation.get("expires_at", ""),
            "idempotency_key": pending_confirmation.get("idempotency_key", ""),
            "token_present": token_present,
        }
    ]


def _build_risk_warning_texts(risk_codes: list[str]) -> list[str]:
    """把 verification.risk_warnings 的 code 翻译成用户可见的中文。"""

    texts: list[str] = []
    for code in risk_codes:
        label = RISK_WARN_LABELS.get(code)
        if label and label not in texts:
            texts.append(label)
    return texts


def _build_citations(
    steps: list[dict[str, Any]],
    step_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """把每个 RAG step 的 citations 汇总成扁平列表。

    结构（保留 step 归属以便前端按 step 分组展示）：
    `{step, ref, doc_id, chunk_id, doc_title, source, section_title}`
    其余字段（content / score 等）不暴露，避免把长正文或内部分数推给前端。
    同一 `(step, doc_id, ref)` 不重复；顺序按 step 声明顺序 + ref 顺序。
    """

    flattened: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for step in steps:
        if step.get("agent") != STEP_AGENT_RAG:
            continue
        step_id = step.get("id", "")
        res = step_results.get(step_id) or {}
        if res.get("status") != STEP_STATUS_SUCCEEDED:
            continue
        for citation in res.get("citations") or []:
            ref = str(citation.get("ref") or "")
            doc_id = str(citation.get("doc_id") or "")
            key = (step_id, doc_id, ref)
            if key in seen:
                continue
            seen.add(key)
            flattened.append(
                {
                    "step": step_id,
                    "ref": ref,
                    "doc_id": doc_id,
                    "chunk_id": citation.get("chunk_id", ""),
                    "doc_title": citation.get("doc_title", ""),
                    "source": citation.get("source", ""),
                    "section_title": citation.get("section_title", ""),
                }
            )
    return flattened


def _collect_step_failures(
    steps: list[dict[str, Any]],
    step_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for step in steps:
        step_id = step.get("id", "")
        res = step_results.get(step_id) or {}
        if res.get("status") == STEP_STATUS_FAILED:
            failures.append(
                {
                    "step": step_id,
                    "agent": step.get("agent", ""),
                    "tool": step.get("tool") or res.get("tool_name") or "",
                    "error": str(res.get("error") or ""),
                }
            )
    return failures


# --------------------------------------------------------------------------
# 答案拼装（按 workflow_status 分支）
# --------------------------------------------------------------------------


def _compose_success_answer(
    steps: list[dict[str, Any]],
    step_results: dict[str, dict[str, Any]],
) -> str:
    """成功路径：按 step 顺序拼接 succeeded step 的 output。

    空输出自动跳过；所有 step 都空输出时返回空字符串，由上层兜底提示。
    """

    pieces: list[str] = []
    for step in steps:
        res = step_results.get(step.get("id", "")) or {}
        if res.get("status") != STEP_STATUS_SUCCEEDED:
            continue
        text = str(res.get("output") or "").strip()
        if text:
            pieces.append(text)
    return "\n".join(pieces)


def _compose_partial_answer(
    steps: list[dict[str, Any]],
    step_results: dict[str, dict[str, Any]],
) -> str:
    """部分成功：成功片段 + 失败归因的中文摘要。"""

    success_text = _compose_success_answer(steps, step_results)
    failures = _collect_step_failures(steps, step_results)
    skipped = [
        step.get("id", "")
        for step in steps
        if (step_results.get(step.get("id", "")) or {}).get("status")
        == STEP_STATUS_SKIPPED
    ]

    lines: list[str] = []
    if success_text:
        lines.append(success_text)
    if failures:
        failure_summary = "、".join(
            f.get("step") or f.get("tool") or "未知步骤" for f in failures
        )
        lines.append(f"以下步骤未能成功执行：{failure_summary}。")
    if skipped:
        lines.append(f"因上游失败，以下步骤未执行：{', '.join(skipped)}。")
    return "\n".join(lines) if lines else COMPOSER_FALLBACK_ALL_FAILED


# --------------------------------------------------------------------------
# Node 主入口
# --------------------------------------------------------------------------


def _emit_answer(state: AgentState, answer: str) -> bool:
    """把最终 answer 作为一次性 chunk 发给上游流式通道。

    之所以不分 token 流：Composer 是模板合成，没有增量产出模型；把整段一次性
    发出，保持和 merge_node 的 SSE 契约一致（前端看到一条 chunk + 一条
    node_completed）。返回值标识"是否真的推过流"。
    """

    stream_callback = get_stream_callback()
    if not callable(stream_callback) or not answer:
        return False
    stream_callback(
        "chunk",
        {
            "request_id": state.get("request_id", ""),
            "session_id": state.get("session_id", "default"),
            "node": NODE_COMPOSER,
            "delta": answer,
        },
    )
    return True


def composer_node(state: AgentState) -> AgentState:
    """Composer 主节点。

    依赖字段：`plan` / `step_results` / `verification` / `workflow_status` /
    `pending_confirmation`。任一缺失视为"最保守回复"而不是抛错，保证在 Planner
    一上来就失败的路径上也能吐出文案。
    """

    plan = state.get("plan") or {}
    steps = list(plan.get("steps") or [])
    step_results: dict[str, dict[str, Any]] = state.get("step_results") or {}
    verification = state.get("verification") or {}
    pending_confirmation = state.get("pending_confirmation") or {}
    workflow_status = state.get("workflow_status", WORKFLOW_STATUS_SUCCEEDED)

    risk_codes = list(verification.get("risk_warnings") or [])
    missing_fields = list(verification.get("missing_fields") or [])

    # --- 分支 1：Planner 失败（plan 为空） ---
    if not steps:
        answer = COMPOSER_FALLBACK_PLAN_FAILED
        claims = list(verification.get("unsupported_claims") or [])
        if claims:
            answer = f"{answer}（失败原因：{'、'.join(claims)}）"
    # --- 分支 2：需要用户确认（side_effect step 短路） ---
    elif workflow_status == WORKFLOW_STATUS_NEED_CONFIRMATION:
        # pipeline 已经在 step.output 里塞了含 token 的完整提示文案；优先透传它，
        # 没有时再回退到通用模板。两者都能让客户端知道"附上 token 再来一次"。
        pending_step_output = ""
        for step in steps:
            res = step_results.get(step.get("id", "")) or {}
            if res.get("status") == STEP_STATUS_NEED_CONFIRMATION:
                pending_step_output = str(res.get("output") or "").strip()
                break
        answer = pending_step_output or COMPOSER_FALLBACK_NEED_CONFIRMATION
    # --- 分支 3：需要澄清（缺参等） ---
    elif workflow_status == WORKFLOW_STATUS_NEED_CLARIFICATION:
        missing_text = "、".join(missing_fields) if missing_fields else "必需信息"
        answer = f"{COMPOSER_FALLBACK_NEED_CLARIFICATION}{missing_text}"
    # --- 分支 4：整体失败 ---
    elif workflow_status == WORKFLOW_STATUS_FAILED:
        # 失败路径也要尽力保留已成功 step 的片段，方便用户继续对话；
        # 没有成功片段时回退到全失败模板。
        answer = (
            _compose_partial_answer(steps, step_results) or COMPOSER_FALLBACK_ALL_FAILED
        )
    # --- 分支 5：部分成功 ---
    elif workflow_status == WORKFLOW_STATUS_PARTIAL:
        answer = _compose_partial_answer(steps, step_results)
    # --- 分支 6：成功（含 succeeded / 未知状态兜底） ---
    else:
        answer = (
            _compose_success_answer(steps, step_results) or COMPOSER_FALLBACK_ALL_FAILED
        )

    # 风险提示统一附加到 answer 末尾，保证任何分支都不会漏掉 UI 徽标。
    risk_texts = _build_risk_warning_texts(risk_codes)
    if risk_texts:
        answer = answer.rstrip() + "\n" + "\n".join(f"- {t}" for t in risk_texts)

    completed_actions = _build_completed_actions(steps, step_results)
    pending_confirmations = _build_pending_confirmations(pending_confirmation)
    citations = _build_citations(steps, step_results)

    composer_output: dict[str, Any] = {
        "answer": answer,
        "completed_actions": completed_actions,
        "pending_confirmations": pending_confirmations,
        "missing_information": missing_fields,
        "citations": citations,
        "risk_warnings": risk_texts,
        "workflow_status": workflow_status,
    }

    streamed = _emit_answer(state, answer)

    next_state: AgentState = {
        "answer": answer,
        "agent_outputs": {COMPOSER_OUTPUT_KEY: composer_output},
        "debug_info": {
            NODE_COMPOSER: {
                "plan_id": state.get("plan_id", ""),
                "workflow_status": workflow_status,
                "completed_action_count": len(completed_actions),
                "pending_confirmation_count": len(pending_confirmations),
                "missing_fields": missing_fields,
                "risk_warnings": risk_texts,
                "streamed_answer": streamed,
                "final_answer_preview": preview(answer, 160),
            }
        },
    }
    if streamed:
        next_state["streamed_answer"] = True

    log_state = {**state, **next_state}
    log_node(
        NODE_COMPOSER,
        log_state,
        extra={
            "planId": state.get("plan_id", ""),
            "workflowStatus": workflow_status,
            "completedActions": len(completed_actions),
            "pendingConfirmations": len(pending_confirmations),
            "finalAnswerPreview": preview(answer, 160),
        },
    )
    return next_state


# 公开给 test / eval 使用的小工具，避免测试重新推导摘要逻辑。
__all__ = [
    "composer_node",
]


# 防 lint 报警（某些辅助常量在主流程里没直接引用，但文件对外语义需要）：
_ = STEP_AGENT_RAG
_ = STEP_AGENT_TOOL

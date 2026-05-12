"""Workflow Executor：顺序驱动 Planner 产出的 plan.steps。

Phase 2 MVP 约束（对齐总设 §6）：
- **单请求同步**：一次 graph 请求内跑完所有 step；跨请求续跑（waiting_user 恢复）
  留给 Phase 2.5。
- **顺序 DAG**：steps 已经被 schema 层约束为"前向依赖"，这里按声明顺序执行就能
  满足 depends_on；不做并发。
- **短路**：任一 step 触发 `need_confirmation` / `failed` 立即停，后续 step 标记
  `skipped`。避免"确认未到就继续写下一张工单"这种失控发散。
- **side_effect 工具走 pipeline**：Planner 决定了 tool+args 后，Executor 直接
  复用 `wrap_side_effect_tool`（匿名拦截、idempotency、confirmation、超时、终态
  写入都在里面），不再经过 LLM function-calling，避免二次 args drift。
- **read_only 工具直接调 impl**：已经被 Planner 选定且参数固定，再走 LLM 只会
  让流程更慢、更不稳定。

不负责：
- 真正的 Verifier（PR-3）：这里只把执行结果写入 `step_results`，由下游节点检查
  args 完整性 / 回答是否有事实支撑。
- Composer（PR-4）：这里生成一个"兜底拼接文案"作为 `agent_outputs` 的 value，
  让当前 MVP 的 merge_node 不至于输出空串；Composer 上线后会覆盖。
"""

from __future__ import annotations

from typing import Any, Callable

from app.auth.context import AuthContext
from app.constants.routes import ROUTE_RAG_AGENT, ROUTE_TOOL_AGENT
from app.constants.workflow import (
    NODE_WORKFLOW_EXECUTOR,
    STEP_AGENT_CHAT,
    STEP_AGENT_RAG,
    STEP_AGENT_TOOL,
    STEP_STATUS_FAILED,
    STEP_STATUS_NEED_CONFIRMATION,
    STEP_STATUS_SKIPPED,
    STEP_STATUS_SUCCEEDED,
    WORKFLOW_STATUS_FAILED,
    WORKFLOW_STATUS_NEED_CONFIRMATION,
    WORKFLOW_STATUS_SUCCEEDED,
)
from app.state import AgentState
from app.tools.metadata import ToolNotRegisteredError
from app.tools.pipeline import (
    SideEffectContext,
    format_result_for_llm,
    wrap_side_effect_tool,
)
from app.utils.logger import log_node, log_warning
from app.workflow.registry import ToolRegistry, default_tool_registry
from app.workflow.tool_args import filter_args_by_spec

# 把 tool_agent 里的 TOOL_IMPLS 作为唯一实现来源，避免执行器维持一份副本。
# tool_agent 同样引用它，二者自然保持一致（常量抽离原则）。
from app.agents.tool_agent import TOOL_IMPLS

# --------------------------------------------------------------------------
# Step 执行
# --------------------------------------------------------------------------


def _resolve_tool(
    step: dict[str, Any], registry: ToolRegistry, auth: AuthContext
) -> tuple[str, Callable[..., Any], bool] | None:
    """返回 (canonical_name, impl, is_side_effect)；任何失败返回 None。

    之所以把鉴权/白名单再次检查一遍：Planner 也会做，但二级防御确保即便
    Planner 跑飞（被注入恶意输出、tests 伪造 plan 等）也不会绕过匿名 × side_effect
    的硬约束。
    """

    raw = step.get("tool") or ""
    try:
        canonical = registry.resolve(raw)
    except ToolNotRegisteredError:
        return None
    meta = registry.get(canonical)
    if auth.anonymous and meta.side_effect:
        # 防御性：匿名不允许执行副作用工具。
        return None
    impl = TOOL_IMPLS.get(canonical)
    if impl is None:
        return None
    return canonical, impl, meta.side_effect


def run_tool_step(
    step: dict[str, Any],
    state: AgentState,
    *,
    registry: ToolRegistry,
    side_ctx: SideEffectContext,
) -> dict[str, Any]:
    """执行单个 tool_agent step。

    返回结构由 `step_results[step_id]` 消费：
    - `status`：step 状态（成功/失败/need_confirmation）
    - `output`：字符串化结果，供 Composer/Verifier 读取
    - `tool_name` / `args`：审计信息
    - `pending_confirmation`：仅在 need_confirmation 时填
    """

    resolved = _resolve_tool(step, registry, side_ctx.auth)
    if resolved is None:
        return {
            "status": STEP_STATUS_FAILED,
            "output": "",
            "error": "tool_not_resolvable",
            "tool_name": step.get("tool", ""),
        }

    canonical, impl, is_side_effect = resolved
    raw_args: dict[str, Any] = dict(step.get("args") or {})
    args, dropped = filter_args_by_spec(canonical, raw_args)
    if dropped:
        # 记 warning 而不是 error：Planner 幻觉的多余参数被静默丢弃是允许的
        # 防御机制；真正缺 required 参数会由 Verifier 判掉。
        log_warning(
            stage=NODE_WORKFLOW_EXECUTOR,
            message="planner args filtered by spec",
            extra={
                "tool_name": canonical,
                "dropped_keys": list(dropped),
                "step_id": step.get("id", ""),
            },
        )

    if is_side_effect:
        meta = registry.get(canonical)
        wrapped = wrap_side_effect_tool(canonical, impl, meta, side_ctx)
        try:
            output = wrapped(**args)
        except Exception as exc:  # noqa: BLE001 — 与 tool_agent 对齐：错误不冒泡
            return {
                "status": STEP_STATUS_FAILED,
                "output": "",
                "error": f"{type(exc).__name__}: {exc}",
                "tool_name": canonical,
                "args": args,
            }

        # pipeline 通过 side_ctx.pending_confirmation 反向回传 need_confirmation 信号。
        if side_ctx.pending_confirmation:
            pending = dict(side_ctx.pending_confirmation)
            # 清空 side_ctx 该字段，避免后续 step 误判（本 MVP 短路，理论上不会有后续）。
            side_ctx.pending_confirmation = None
            return {
                "status": STEP_STATUS_NEED_CONFIRMATION,
                "output": output,
                "tool_name": canonical,
                "args": args,
                "pending_confirmation": pending,
            }
        return {
            "status": STEP_STATUS_SUCCEEDED,
            "output": output,
            "tool_name": canonical,
            "args": args,
        }

    # read_only 工具：直接调 impl，无 pipeline 开销。
    try:
        raw_result = impl(**args)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": STEP_STATUS_FAILED,
            "output": "",
            "error": f"{type(exc).__name__}: {exc}",
            "tool_name": canonical,
            "args": args,
        }
    return {
        "status": STEP_STATUS_SUCCEEDED,
        "output": format_result_for_llm(raw_result),
        "tool_name": canonical,
        "args": args,
    }


def run_rag_step(step: dict[str, Any], state: AgentState) -> dict[str, Any]:
    """执行单个 rag_agent step。

    做法：合成一个临时 state，把 `step.query` 作为最新 user 消息，调用
    `rag_agent_node`；只吃它的 `answer`，不继承它写回 state 的其他副作用
    （memory_hits、context 等）避免污染 workflow 主 state。
    """

    # 延迟 import，避免模块循环依赖（rag → state → executor）。
    from app.agents.rag_agent import rag_agent_node

    query = step.get("query") or ""
    synth_state: AgentState = {
        **state,
        "messages": [{"role": "user", "content": query}],
        # 清掉上一轮的流式回调：workflow 场景不希望 RAG 把增量直接吐给客户端，
        # 否则客户端会看到多段混乱的 token。最终只由 Composer 输出。
        "stream_callback": None,
    }
    try:
        sub_result = rag_agent_node(synth_state)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": STEP_STATUS_FAILED,
            "output": "",
            "error": f"{type(exc).__name__}: {exc}",
            "query": query,
        }
    answer = sub_result.get("answer") or ""
    return {
        "status": STEP_STATUS_SUCCEEDED,
        "output": answer,
        "query": query,
    }


def run_chat_step(step: dict[str, Any], state: AgentState) -> dict[str, Any]:
    """执行单个 chat_agent step。

    Phase 2 MVP 下 chat_agent step 很少出现在 plan（Planner 倾向拆到 tool/rag）。
    这里不引入额外 LLM 调用，直接把 `step.purpose` 作为占位输出。Composer
    拿到后会按 plan.compose_goal 重新整合，不依赖这段文案的措辞质量。
    """

    del state  # 暂未消费；保留参数以便后续接入 chat LLM 不改签名。
    purpose = step.get("purpose") or ""
    return {
        "status": STEP_STATUS_SUCCEEDED,
        "output": purpose,
    }


# --------------------------------------------------------------------------
# Node 主入口
# --------------------------------------------------------------------------


_STEP_DISPATCH: dict[str, str] = {
    STEP_AGENT_TOOL: ROUTE_TOOL_AGENT,
    STEP_AGENT_RAG: ROUTE_RAG_AGENT,
    STEP_AGENT_CHAT: "chat_agent",
}


def _compose_fallback_answer(
    step_results: dict[str, dict[str, Any]],
    steps: list[dict[str, Any]],
) -> str:
    """在 Composer 上线前，给 merge_node 一个可读的兜底回答。

    只拼 succeeded step 的 `output`，need_confirmation 另行提示。失败 step
    不进回答正文，避免把错误信息泄露给最终用户（留给 Composer 统一处理）。
    """

    pieces: list[str] = []
    for step in steps:
        res = step_results.get(step["id"]) or {}
        if res.get("status") != STEP_STATUS_SUCCEEDED:
            continue
        output = str(res.get("output") or "").strip()
        if output:
            pieces.append(output)
    return "\n".join(pieces)


def workflow_executor_node(
    state: AgentState,
    *,
    registry: ToolRegistry | None = None,
) -> AgentState:
    """Workflow Executor 主节点。

    Planner 上游已经把 plan 写进 state；这里拿 steps 按顺序跑。Planner 失败
    （plan 空）时原地短路返回：Composer/merge 会基于 `verification` 生成
    拒绝文案，不由 Executor 自行合成。
    """

    registry = registry or default_tool_registry
    plan = state.get("plan") or {}
    steps = list(plan.get("steps") or [])
    plan_id = state.get("plan_id", "")

    # Planner 已 fail-closed：不重复写 verification，只记录一次 debug_info，
    # 确保 trace 能看到"executor 被调用但没东西可执行"。
    if not steps:
        return {
            "workflow_status": state.get("workflow_status", WORKFLOW_STATUS_FAILED),
            "step_results": {},
            "debug_info": {
                NODE_WORKFLOW_EXECUTOR: {
                    "plan_id": plan_id,
                    "status": "noop",
                    "reason": "empty_plan",
                }
            },
        }

    auth = state["auth"]
    side_ctx = SideEffectContext(
        auth=auth,
        session_id=state.get("session_id", "default"),
        request_id=state.get("request_id", ""),
        confirmation_token=state.get("confirmation_token", "") or "",
    )

    step_results: dict[str, dict[str, Any]] = {}
    final_status = WORKFLOW_STATUS_SUCCEEDED
    pending_confirmation: dict[str, Any] = {}
    short_circuited = False

    for idx, step in enumerate(steps):
        if short_circuited:
            # 已触发 need_confirmation / failed：剩余 step 标记 skipped。
            step_results[step["id"]] = {
                "status": STEP_STATUS_SKIPPED,
                "output": "",
                "reason": "upstream_short_circuit",
            }
            continue

        agent = step.get("agent")
        if agent == STEP_AGENT_TOOL:
            result = run_tool_step(step, state, registry=registry, side_ctx=side_ctx)
        elif agent == STEP_AGENT_RAG:
            result = run_rag_step(step, state)
        elif agent == STEP_AGENT_CHAT:
            result = run_chat_step(step, state)
        else:  # 防御：schema 已限制 agent ∈ STEP_AGENTS
            result = {
                "status": STEP_STATUS_FAILED,
                "output": "",
                "error": f"unknown_agent:{agent}",
            }

        step_results[step["id"]] = result

        if result["status"] == STEP_STATUS_NEED_CONFIRMATION:
            final_status = WORKFLOW_STATUS_NEED_CONFIRMATION
            pending_confirmation = result.get("pending_confirmation", {}) or {}
            short_circuited = True
            log_warning(
                NODE_WORKFLOW_EXECUTOR,
                "workflow short-circuited at need_confirmation",
                {
                    "plan_id": plan_id,
                    "step_id": step["id"],
                    "step_index": idx,
                    "tool_name": result.get("tool_name"),
                },
            )
            continue
        if result["status"] == STEP_STATUS_FAILED:
            final_status = WORKFLOW_STATUS_FAILED
            short_circuited = True
            log_warning(
                NODE_WORKFLOW_EXECUTOR,
                "workflow step failed; short-circuit",
                {
                    "plan_id": plan_id,
                    "step_id": step["id"],
                    "step_index": idx,
                    "error": result.get("error"),
                },
            )
            continue

    fallback_answer = _compose_fallback_answer(step_results, steps)
    if final_status == WORKFLOW_STATUS_NEED_CONFIRMATION and pending_confirmation:
        # 覆盖 answer 为 pipeline 的 need_confirmation 提示文案，
        # 与 tool_agent 行为对齐（客户端看到明确的"需要确认"）。
        fallback_answer = str(
            step_results.get(
                next(
                    (
                        sid
                        for sid, r in step_results.items()
                        if r.get("status") == STEP_STATUS_NEED_CONFIRMATION
                    ),
                    "",
                ),
                {},
            ).get("output", "")
            or fallback_answer
        )

    next_state: AgentState = {
        "workflow_status": final_status,
        "step_results": step_results,
        "tool_executions": list(side_ctx.executions),
        "pending_confirmation": pending_confirmation,
        "agent_outputs": {NODE_WORKFLOW_EXECUTOR: fallback_answer},
        "debug_info": {
            NODE_WORKFLOW_EXECUTOR: {
                "plan_id": plan_id,
                "status": final_status,
                "step_count": len(steps),
                "step_results": step_results,
                "tool_executions": list(side_ctx.executions),
                "pending_confirmation": pending_confirmation,
            }
        },
    }

    log_state = {**state, **next_state}
    log_node(
        NODE_WORKFLOW_EXECUTOR,
        log_state,
        extra={
            "planId": plan_id,
            "workflowStatus": final_status,
            "stepCount": len(steps),
            "shortCircuited": short_circuited,
        },
    )
    return next_state


__all__ = [
    "run_chat_step",
    "run_rag_step",
    "run_tool_step",
    "workflow_executor_node",
]

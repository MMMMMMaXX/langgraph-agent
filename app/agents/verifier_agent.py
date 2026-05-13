"""Verifier Agent：Workflow Executor 执行结果的规则化把关（Phase 2 PR-3）。

职责边界（对齐 `docs/phase2-workflow-agent.md` §6）：

- **纯规则驱动**：MVP 不在在线链路跑 LLM claim judge，避免把 Verifier 变成第二
  个 Planner。所有检查项都是对 plan + step_results + auth + tool metadata 做
  确定性布尔判断。
- **不重新执行**：Verifier 只读 Executor 写入的 `step_results` / `tool_executions`，
  不再触碰 side_effect 工具；任何"重新跑一次以自证"的想法都违反幂等原则。
- **分层与 Planner/Executor 的错**：
  - Planner 阶段失败（`plan={}`、`verification.unsupported_claims` 已有 ERR_PLAN_*）
    → Verifier 直接原样透传结论，只补充 `risk_warnings=[]`，避免把"计划生成失败"
    误报成"执行参数不全"。
  - Executor 汇报 `need_confirmation` → Verifier 保持 need_confirmation，累加
    `side_effect_requires_user_confirmation` 风险提示。
  - Executor 汇报 `failed` → 直接沉淀失败 step 的 `verify_step_failed` claim。
- **Status 合并优先级**（写回 `workflow_status`）：
  `failed` > `need_confirmation` > `need_clarification` > `partial` > `succeeded`
  取两路（Executor 上报 + Verifier 结论）最严重的那档。这样 Executor 成功但
  Verifier 发现参数缺失时，会降级到 `need_clarification`，Composer 可据此回到
  用户那里确认。

检查项（仅在成功/部分成功路径上跑完整规则）：

1. `verify_tool_unauthorized`：tool_agent step 引用了当前 auth 看不到的 tool
   （匿名 × side_effect 是典型）。即便 Executor 把这种 step 标了 failed，
    Verifier 也要把原因写进 unsupported_claims，供 Composer 告知用户。
2. `verify_missing_args`：tool step 的 required 参数缺失。从 tool_agent.TOOLS
    的 function-calling schema 派生，保持唯一真相源。
3. `verify_step_failed`：任何 step_results[step.id] 为 failed 的条目，都生成
    一条 claim，方便 Composer 按 step id 生成归因文案。
4. `side_effect_requires_user_confirmation`：所有 side_effect step（不管
   succeeded 还是 need_confirmation）都挂这条风险提示，让 UI 层能一致提示
   "此操作有副作用"。
5. `high_risk_tool_invoked`：命中 `risk_level >= medium` 的工具就挂这条。
"""

from __future__ import annotations

from typing import Any

from app.constants.multi_hop import (
    MULTI_HOP_STEP_ID,
    RISK_WARN_MULTI_HOP_COVERAGE,
)
from app.constants.tool_safety import RISK_LEVEL_HIGH, RISK_LEVEL_MEDIUM
from app.constants.workflow import (
    ERR_VERIFY_MISSING_ARGS,
    ERR_VERIFY_STEP_FAILED,
    ERR_VERIFY_TOOL_UNAUTHORIZED,
    NODE_VERIFIER,
    RISK_WARN_HIGH_RISK_TOOL,
    RISK_WARN_RAG_MISSING_CITATION,
    RISK_WARN_SIDE_EFFECT_CONFIRMED,
    STEP_AGENT_RAG,
    STEP_AGENT_TOOL,
    STEP_STATUS_FAILED,
    STEP_STATUS_NEED_CONFIRMATION,
    STEP_STATUS_SUCCEEDED,
    TASK_TYPE_MULTI_HOP_RAG,
    VERIFICATION_STATUS_FAILED,
    VERIFICATION_STATUS_NEED_CLARIFICATION,
    VERIFICATION_STATUS_NEED_CONFIRMATION,
    VERIFICATION_STATUS_PASS,
    WORKFLOW_STATUS_FAILED,
    WORKFLOW_STATUS_NEED_CLARIFICATION,
    WORKFLOW_STATUS_NEED_CONFIRMATION,
    WORKFLOW_STATUS_PARTIAL,
    WORKFLOW_STATUS_PENDING,
    WORKFLOW_STATUS_RUNNING,
    WORKFLOW_STATUS_SUCCEEDED,
)
from app.state import AgentState
from app.tools.metadata import ToolNotRegisteredError
from app.utils.logger import log_node
from app.workflow.registry import ToolRegistry, default_tool_registry
from app.workflow.tool_args import REQUIRED_ARGS as _REQUIRED_ARGS

# --------------------------------------------------------------------------
# Status 合并
# --------------------------------------------------------------------------

# 数字越大越"严重"；合并时取 max。这份表是 workflow_status 与
# verification.status 共享的严重度等级，Verifier 再按结果回写 workflow_status。
_STATUS_SEVERITY: dict[str, int] = {
    WORKFLOW_STATUS_SUCCEEDED: 0,
    WORKFLOW_STATUS_PENDING: 0,
    WORKFLOW_STATUS_RUNNING: 0,
    VERIFICATION_STATUS_PASS: 0,
    WORKFLOW_STATUS_PARTIAL: 1,
    WORKFLOW_STATUS_NEED_CLARIFICATION: 2,
    VERIFICATION_STATUS_NEED_CLARIFICATION: 2,
    WORKFLOW_STATUS_NEED_CONFIRMATION: 3,
    VERIFICATION_STATUS_NEED_CONFIRMATION: 3,
    WORKFLOW_STATUS_FAILED: 4,
    VERIFICATION_STATUS_FAILED: 4,
}


def _merge_workflow_status(executor_status: str, verification_status: str) -> str:
    """按严重度 max 合并两个状态字符串。

    Verifier 的结论用 verification.* 常量；Executor 用 workflow_status.* 常量。
    两类常量在"严重度"维度上一一对应，合并后统一落到 workflow_status 命名空间。
    """

    left = _STATUS_SEVERITY.get(executor_status, 0)
    right = _STATUS_SEVERITY.get(verification_status, 0)
    winner = max(left, right)

    # 按严重度反查 workflow_status 侧的规范值，避免把 verification.* 写进
    # workflow_status 字段（两套枚举不应互串）。
    if winner >= _STATUS_SEVERITY[WORKFLOW_STATUS_FAILED]:
        return WORKFLOW_STATUS_FAILED
    if winner >= _STATUS_SEVERITY[WORKFLOW_STATUS_NEED_CONFIRMATION]:
        return WORKFLOW_STATUS_NEED_CONFIRMATION
    if winner >= _STATUS_SEVERITY[WORKFLOW_STATUS_NEED_CLARIFICATION]:
        return WORKFLOW_STATUS_NEED_CLARIFICATION
    if winner >= _STATUS_SEVERITY[WORKFLOW_STATUS_PARTIAL]:
        return WORKFLOW_STATUS_PARTIAL
    return WORKFLOW_STATUS_SUCCEEDED


# --------------------------------------------------------------------------
# 检查规则
# --------------------------------------------------------------------------


def _check_tool_step(
    step: dict[str, Any],
    step_result: dict[str, Any],
    *,
    registry: ToolRegistry,
    auth_anonymous: bool,
    missing_fields: list[str],
    unsupported_claims: list[str],
    risk_warnings: list[str],
) -> None:
    """对单个 tool_agent step 执行规则检查，结果累加到传入的 list。"""

    raw_tool = step.get("tool") or ""
    step_id = step.get("id", "")

    # 1) 工具是否可解析 / 是否越权：二次防御 Planner/Executor 可能漏的。
    try:
        canonical = registry.resolve(raw_tool)
        meta = registry.get(canonical)
    except ToolNotRegisteredError:
        # 未注册工具：Planner 已有 ERR_PLAN_UNKNOWN_TOOL，这里不重复登记，
        # 只在当前 step 维度补一条 step_failed 方便按 step 追踪。
        unsupported_claims.append(f"{ERR_VERIFY_STEP_FAILED}:{step_id}")
        return

    if auth_anonymous and meta.side_effect:
        unsupported_claims.append(f"{ERR_VERIFY_TOOL_UNAUTHORIZED}:{step_id}")
        # 越权与风险提示并行：让 Composer 同时看到"被拦"和"这是副作用工具"。
        risk_warnings.append(RISK_WARN_SIDE_EFFECT_CONFIRMED)

    # 2) 必需参数齐不齐。
    required = _REQUIRED_ARGS.get(canonical, ())
    provided = step.get("args") or {}
    for key in required:
        if key not in provided or provided[key] in ("", None):
            missing_fields.append(f"{step_id}.{key}")

    # 3) 风险等级：medium/high 都要提示。
    if meta.risk_level in (RISK_LEVEL_MEDIUM, RISK_LEVEL_HIGH):
        risk_warnings.append(RISK_WARN_HIGH_RISK_TOOL)

    # 4) side_effect 工具：只要 Executor 真的跑进去了（succeeded/need_confirmation），
    #    一律挂 "需要用户确认" 风险提示；Composer 据此让 UI 显示副作用徽标。
    if meta.side_effect and step_result.get("status") in (
        STEP_STATUS_SUCCEEDED,
        STEP_STATUS_NEED_CONFIRMATION,
    ):
        risk_warnings.append(RISK_WARN_SIDE_EFFECT_CONFIRMED)


def _check_step_result(
    step: dict[str, Any],
    step_result: dict[str, Any],
    *,
    unsupported_claims: list[str],
) -> None:
    """把执行层 failed 的 step 翻译成 verify_step_failed claim。"""

    if step_result.get("status") == STEP_STATUS_FAILED:
        step_id = step.get("id", "")
        unsupported_claims.append(f"{ERR_VERIFY_STEP_FAILED}:{step_id}")


def _check_rag_step(
    step: dict[str, Any],
    step_result: dict[str, Any],
    *,
    risk_warnings: list[str],
) -> None:
    """RAG step 证据校验。

    规则：RAG step 成功时，如果 `doc_used=True` 但没有 citations（或 citations
    被丢掉），说明答案没有可追溯的来源——这是质量信号（而非硬错误），
    挂 `rag_step_missing_citation` 让 Composer 在回答尾部追加提示，同时 eval
    可以据此聚合"缺证据回答率"。
    纯 memory 回复（doc_used=False）不触发：没命中文档时本来就不需要引用。
    """

    del step  # 参数保留以对齐其他 _check_* 签名
    if step_result.get("status") != STEP_STATUS_SUCCEEDED:
        return
    if not step_result.get("doc_used"):
        return
    citations = step_result.get("citations") or []
    if not citations:
        risk_warnings.append(RISK_WARN_RAG_MISSING_CITATION)


def _check_multi_hop_coverage(
    step_results: dict[str, dict[str, Any]],
    *,
    risk_warnings: list[str],
) -> None:
    """Multi-hop 覆盖度校验（Phase 3 PR-4）。

    multi_hop_node 在 pseudo-step 的 `meta.missing_coverage_sq_ids` 里
    登记"零 chunk 的 sq_id 列表"。Verifier 只做 presence 检查——faithfulness
    之类的语义评测交给离线 eval，在线链路不跑 LLM judge。

    不覆盖 "doc_used=False" 场景：multi-hop 触发路径本来就是文档依赖，
    没命中证据属于 evidence_empty 失败链路而非缺引用的降级。
    """

    mh = step_results.get(MULTI_HOP_STEP_ID) or {}
    meta = mh.get("meta") or {}
    if meta.get("missing_coverage_sq_ids"):
        risk_warnings.append(RISK_WARN_MULTI_HOP_COVERAGE)


def _derive_verification_status(
    *,
    executor_status: str,
    missing_fields: list[str],
    unsupported_claims: list[str],
) -> str:
    """从检查结果推导 verification.status。

    Executor 汇报已经反映了"整体卡在哪一档"；Verifier 再看自己发现的问题向上提档。
    """

    if executor_status == WORKFLOW_STATUS_FAILED:
        return VERIFICATION_STATUS_FAILED
    # Executor 要求确认：Verifier 要么原样通过（视作 pass 但上层合并会还原为 need_confirmation），
    # 要么因为检查失败升级到 failed。
    if unsupported_claims:
        return VERIFICATION_STATUS_FAILED
    if missing_fields:
        return VERIFICATION_STATUS_NEED_CLARIFICATION
    if executor_status == WORKFLOW_STATUS_NEED_CONFIRMATION:
        return VERIFICATION_STATUS_NEED_CONFIRMATION
    return VERIFICATION_STATUS_PASS


# --------------------------------------------------------------------------
# Node 主入口
# --------------------------------------------------------------------------


def verifier_node(
    state: AgentState,
    *,
    registry: ToolRegistry | None = None,
) -> AgentState:
    """Verifier 主节点。

    - Planner fail-closed 时（`plan={}`）：直接透传已有 verification，只保证
      字段结构完整，不做进一步检查。
    - 其它情况：按 step 跑规则检查，累加 verification 字段并合并 workflow_status。
    """

    registry = registry or default_tool_registry
    plan = state.get("plan") or {}
    steps = list(plan.get("steps") or [])
    plan_id = state.get("plan_id", "")
    executor_status = state.get("workflow_status", WORKFLOW_STATUS_SUCCEEDED)
    step_results: dict[str, dict[str, Any]] = state.get("step_results") or {}
    auth = state["auth"]

    # Planner 失败：保留其已写入的 verification，不再执行规则（没有 steps 可查）。
    if not steps:
        existing = dict(state.get("verification") or {})
        existing.setdefault("status", VERIFICATION_STATUS_FAILED)
        existing.setdefault("missing_fields", [])
        existing.setdefault("unsupported_claims", [])
        existing.setdefault("risk_warnings", [])

        next_state: AgentState = {
            "verification": existing,
            "workflow_status": executor_status or WORKFLOW_STATUS_FAILED,
            "debug_info": {
                NODE_VERIFIER: {
                    "plan_id": plan_id,
                    "status": "noop",
                    "reason": "empty_plan",
                    "verification": existing,
                }
            },
        }
        log_state = {**state, **next_state}
        log_node(
            NODE_VERIFIER,
            log_state,
            extra={
                "planId": plan_id,
                "verificationStatus": existing["status"],
                "noop": True,
            },
        )
        return next_state

    missing_fields: list[str] = []
    unsupported_claims: list[str] = []
    risk_warnings: list[str] = []

    for step in steps:
        step_id = step.get("id", "")
        step_result = step_results.get(step_id) or {}

        if step.get("agent") == STEP_AGENT_TOOL:
            _check_tool_step(
                step,
                step_result,
                registry=registry,
                auth_anonymous=auth.anonymous,
                missing_fields=missing_fields,
                unsupported_claims=unsupported_claims,
                risk_warnings=risk_warnings,
            )
        elif step.get("agent") == STEP_AGENT_RAG:
            _check_rag_step(step, step_result, risk_warnings=risk_warnings)
        # chat step 当前无规则，留空以便未来扩展；仍然跑 step_result 层失败检查。
        _check_step_result(step, step_result, unsupported_claims=unsupported_claims)

    # Phase 3 PR-4：multi-hop 直达路径（task_type = multi_hop_rag）没有
    # 常规 plan.steps，Verifier 按 task_type 直接看 pseudo-step 的 meta。
    # 这条检查对 PR-6 workflow 嵌套路径同样生效：只要 plan.task_type 是
    # multi_hop_rag，coverage 缺失就一定要提示。
    if plan.get("task_type") == TASK_TYPE_MULTI_HOP_RAG:
        _check_multi_hop_coverage(step_results, risk_warnings=risk_warnings)

    # 去重但保留首次出现顺序，避免 Composer 看到重复文案。
    dedup_claims = list(dict.fromkeys(unsupported_claims))
    dedup_missing = list(dict.fromkeys(missing_fields))
    dedup_risks = list(dict.fromkeys(risk_warnings))

    verification_status = _derive_verification_status(
        executor_status=executor_status,
        missing_fields=dedup_missing,
        unsupported_claims=dedup_claims,
    )

    verification = {
        "status": verification_status,
        "missing_fields": dedup_missing,
        "unsupported_claims": dedup_claims,
        "risk_warnings": dedup_risks,
    }

    final_status = _merge_workflow_status(executor_status, verification_status)

    next_state = {
        "verification": verification,
        "workflow_status": final_status,
        "debug_info": {
            NODE_VERIFIER: {
                "plan_id": plan_id,
                "executor_status": executor_status,
                "verification": verification,
                "workflow_status": final_status,
                "step_count": len(steps),
            }
        },
    }

    log_state = {**state, **next_state}
    log_node(
        NODE_VERIFIER,
        log_state,
        extra={
            "planId": plan_id,
            "verificationStatus": verification_status,
            "workflowStatus": final_status,
            "missingFieldsCount": len(dedup_missing),
            "claimsCount": len(dedup_claims),
            "risksCount": len(dedup_risks),
        },
    )
    return next_state


__all__ = ["verifier_node"]

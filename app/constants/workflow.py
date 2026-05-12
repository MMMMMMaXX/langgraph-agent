"""Workflow Agent 相关常量。

Phase 2 MVP：Planner / Executor / Verifier / Composer 共用。所有对外暴露的
状态枚举与路由常量都集中在此，避免各模块维持副本。

命名约定：
- `WORKFLOW_STATUS_*` / `STEP_STATUS_*`：对齐总设 §8.1 的状态机。内部代码/trace
  只用这些枚举；Composer 负责把 `succeeded` 对外映射成 "completed" 文案。
- `ROUTE_WORKFLOW`：Supervisor 路由决策值；与 `app/constants/routes.py` 的既有
  `ROUTE_*` 保持风格一致。
"""

from __future__ import annotations

from typing import Final

# ---- Supervisor 路由 ----------------------------------------------------

# Phase 2：Supervisor 把多步编排任务路由到 workflow 支路（入口是 planner_node）。
ROUTE_WORKFLOW: Final[str] = "workflow_agent"

# ---- LangGraph 节点名 ---------------------------------------------------

NODE_PLANNER: Final[str] = "planner"
NODE_WORKFLOW_EXECUTOR: Final[str] = "workflow_executor"
NODE_VERIFIER: Final[str] = "verifier"
NODE_COMPOSER: Final[str] = "composer"

# ---- Plan / Step 规模限制 -----------------------------------------------

# Plan 最多允许的 step 数。Planner 超过直接拒，避免 LLM 失控放大任务。
MAX_PLAN_STEPS: Final[int] = 6

# step id 规范：s1, s2 ...
STEP_ID_PREFIX: Final[str] = "s"

# ---- Plan step 允许的 agent --------------------------------------------

# 注意：不含 composer。Composer 是末端独立节点，不作为 plan step，
# 避免 "plan 里声明 composer + 末端再跑 Composer" 的双执行歧义。
STEP_AGENT_TOOL: Final[str] = "tool_agent"
STEP_AGENT_RAG: Final[str] = "rag_agent"
STEP_AGENT_CHAT: Final[str] = "chat_agent"

STEP_AGENTS: Final[tuple[str, ...]] = (
    STEP_AGENT_TOOL,
    STEP_AGENT_RAG,
    STEP_AGENT_CHAT,
)

# ---- Step / Workflow 状态枚举（对齐总设 §8.1）---------------------------

STEP_STATUS_PENDING: Final[str] = "pending"
STEP_STATUS_RUNNING: Final[str] = "running"
STEP_STATUS_WAITING_USER: Final[str] = "waiting_user"
STEP_STATUS_NEED_CONFIRMATION: Final[str] = "need_confirmation"
STEP_STATUS_SUCCEEDED: Final[str] = "succeeded"
STEP_STATUS_FAILED: Final[str] = "failed"
STEP_STATUS_SKIPPED: Final[str] = "skipped"
STEP_STATUS_DEDUPLICATED: Final[str] = "deduplicated"

VALID_STEP_STATUSES: Final[tuple[str, ...]] = (
    STEP_STATUS_PENDING,
    STEP_STATUS_RUNNING,
    STEP_STATUS_WAITING_USER,
    STEP_STATUS_NEED_CONFIRMATION,
    STEP_STATUS_SUCCEEDED,
    STEP_STATUS_FAILED,
    STEP_STATUS_SKIPPED,
    STEP_STATUS_DEDUPLICATED,
)

WORKFLOW_STATUS_PENDING: Final[str] = "pending"
WORKFLOW_STATUS_RUNNING: Final[str] = "running"
WORKFLOW_STATUS_WAITING_USER: Final[str] = "waiting_user"
WORKFLOW_STATUS_NEED_CONFIRMATION: Final[str] = "need_confirmation"
WORKFLOW_STATUS_NEED_CLARIFICATION: Final[str] = "need_clarification"
WORKFLOW_STATUS_SUCCEEDED: Final[str] = "succeeded"
WORKFLOW_STATUS_PARTIAL: Final[str] = "partial"
WORKFLOW_STATUS_FAILED: Final[str] = "failed"

VALID_WORKFLOW_STATUSES: Final[tuple[str, ...]] = (
    WORKFLOW_STATUS_PENDING,
    WORKFLOW_STATUS_RUNNING,
    WORKFLOW_STATUS_WAITING_USER,
    WORKFLOW_STATUS_NEED_CONFIRMATION,
    WORKFLOW_STATUS_NEED_CLARIFICATION,
    WORKFLOW_STATUS_SUCCEEDED,
    WORKFLOW_STATUS_PARTIAL,
    WORKFLOW_STATUS_FAILED,
)

# ---- Planner 错误码（写进 verification.unsupported_claims）-------------

ERR_PLAN_SCHEMA_INVALID: Final[str] = "plan_schema_invalid"
ERR_PLAN_LLM_FAILED: Final[str] = "plan_llm_failed"
ERR_PLAN_TOOL_UNAUTHORIZED: Final[str] = "plan_tool_unauthorized"
ERR_PLAN_UNKNOWN_TOOL: Final[str] = "plan_unknown_tool"
ERR_PLAN_ARGS_INVALID: Final[str] = "plan_args_invalid"
ERR_PLAN_DAG_CYCLE: Final[str] = "plan_dag_cycle"
ERR_PLAN_STEP_LIMIT: Final[str] = "plan_step_limit_exceeded"

# ---- Verifier 结构化输出（`state["verification"]`）----------------------

# verification.status 枚举（对齐总设 §6）。与 workflow_status 有概念重叠，
# 但职责不同：verification.status 只反映"本次检查结论"，工作流整体状态由
# verifier_node 按优先级合并（failed > need_confirmation > need_clarification
# > partial > succeeded）后写回 workflow_status。
VERIFICATION_STATUS_PASS: Final[str] = "pass"
VERIFICATION_STATUS_NEED_CLARIFICATION: Final[str] = "need_clarification"
VERIFICATION_STATUS_NEED_CONFIRMATION: Final[str] = "need_confirmation"
VERIFICATION_STATUS_FAILED: Final[str] = "failed"

VALID_VERIFICATION_STATUSES: Final[tuple[str, ...]] = (
    VERIFICATION_STATUS_PASS,
    VERIFICATION_STATUS_NEED_CLARIFICATION,
    VERIFICATION_STATUS_NEED_CONFIRMATION,
    VERIFICATION_STATUS_FAILED,
)

# risk_warnings 固定取值（避免散落拼写）。Composer 会按这些 code 映射到
# 用户可见文案，未来扩展 code 时必须在此登记。
RISK_WARN_SIDE_EFFECT_CONFIRMED: Final[str] = "side_effect_requires_user_confirmation"
RISK_WARN_HIGH_RISK_TOOL: Final[str] = "high_risk_tool_invoked"

# Composer 用于把 risk_warning code 翻译成用户可见的中文提示。集中在此避免
# 文案散落；Composer、eval、前端展示共享同一份映射。
RISK_WARN_LABELS: Final[dict[str, str]] = {
    RISK_WARN_SIDE_EFFECT_CONFIRMED: "此操作具有副作用，需要您的二次确认。",
    RISK_WARN_HIGH_RISK_TOOL: "本次操作涉及中/高风险工具，请复核。",
}

# ---- Verifier 错误码（写进 verification.unsupported_claims / missing_fields）

ERR_VERIFY_MISSING_ARGS: Final[str] = "verify_missing_args"
ERR_VERIFY_TOOL_UNAUTHORIZED: Final[str] = "verify_tool_unauthorized"
ERR_VERIFY_STEP_FAILED: Final[str] = "verify_step_failed"

# ---- Composer 输出 schema（写进 `state["agent_outputs"]["composer"]`）----

# agent_outputs 的 key；保持字符串常量，避免把 NODE_COMPOSER 这个"节点名"
# 与"输出键"混用——虽然当前相等，但职责不同，分开写让后续改名互不影响。
COMPOSER_OUTPUT_KEY: Final[str] = "composer"

# Composer 兜底文案（plan 为空 / 全部 step 失败时使用）。集中常量便于 eval 断言。
COMPOSER_FALLBACK_PLAN_FAILED: Final[str] = (
    "抱歉，我没能生成安全可执行的计划，请稍后重试或换一种表述。"
)
COMPOSER_FALLBACK_ALL_FAILED: Final[str] = (
    "很抱歉，本次任务未能完成，部分步骤执行失败。"
)
COMPOSER_FALLBACK_NEED_CONFIRMATION: Final[str] = "本次操作需要您的确认。"
COMPOSER_FALLBACK_NEED_CLARIFICATION: Final[str] = "请补充以下信息后再试："

# ---- intent（Supervisor 暴露到 state["intent"] 的字符串）---------------

INTENT_WORKFLOW: Final[str] = "workflow"


__all__ = [
    "COMPOSER_FALLBACK_ALL_FAILED",
    "COMPOSER_FALLBACK_NEED_CLARIFICATION",
    "COMPOSER_FALLBACK_NEED_CONFIRMATION",
    "COMPOSER_FALLBACK_PLAN_FAILED",
    "COMPOSER_OUTPUT_KEY",
    "ERR_PLAN_ARGS_INVALID",
    "ERR_PLAN_DAG_CYCLE",
    "ERR_PLAN_LLM_FAILED",
    "ERR_PLAN_SCHEMA_INVALID",
    "ERR_PLAN_STEP_LIMIT",
    "ERR_PLAN_TOOL_UNAUTHORIZED",
    "ERR_PLAN_UNKNOWN_TOOL",
    "ERR_VERIFY_MISSING_ARGS",
    "ERR_VERIFY_STEP_FAILED",
    "ERR_VERIFY_TOOL_UNAUTHORIZED",
    "INTENT_WORKFLOW",
    "MAX_PLAN_STEPS",
    "NODE_COMPOSER",
    "NODE_PLANNER",
    "NODE_VERIFIER",
    "NODE_WORKFLOW_EXECUTOR",
    "RISK_WARN_HIGH_RISK_TOOL",
    "RISK_WARN_LABELS",
    "RISK_WARN_SIDE_EFFECT_CONFIRMED",
    "ROUTE_WORKFLOW",
    "STEP_AGENTS",
    "STEP_AGENT_CHAT",
    "STEP_AGENT_RAG",
    "STEP_AGENT_TOOL",
    "STEP_ID_PREFIX",
    "STEP_STATUS_DEDUPLICATED",
    "STEP_STATUS_FAILED",
    "STEP_STATUS_NEED_CONFIRMATION",
    "STEP_STATUS_PENDING",
    "STEP_STATUS_RUNNING",
    "STEP_STATUS_SKIPPED",
    "STEP_STATUS_SUCCEEDED",
    "STEP_STATUS_WAITING_USER",
    "VALID_STEP_STATUSES",
    "VALID_VERIFICATION_STATUSES",
    "VALID_WORKFLOW_STATUSES",
    "VERIFICATION_STATUS_FAILED",
    "VERIFICATION_STATUS_NEED_CLARIFICATION",
    "VERIFICATION_STATUS_NEED_CONFIRMATION",
    "VERIFICATION_STATUS_PASS",
    "WORKFLOW_STATUS_FAILED",
    "WORKFLOW_STATUS_NEED_CLARIFICATION",
    "WORKFLOW_STATUS_NEED_CONFIRMATION",
    "WORKFLOW_STATUS_PARTIAL",
    "WORKFLOW_STATUS_PENDING",
    "WORKFLOW_STATUS_RUNNING",
    "WORKFLOW_STATUS_SUCCEEDED",
    "WORKFLOW_STATUS_WAITING_USER",
]

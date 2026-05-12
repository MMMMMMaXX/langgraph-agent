"""Planner Agent：把用户请求拆成结构化 plan（Phase 2 MVP）。

职责边界（有意保持窄）：
- 只负责：拿到 user message + AuthContext → 调 LLM → 产出 Plan 或失败信号。
- 不做：step 执行、side_effect 安全检查、确认 token 重放。Executor/Verifier/Composer
  分别承担。
- 失败一律 **fail-closed**：LLM 调用失败 / JSON schema 不合法 / 引用未注册工具 →
  `workflow_status="failed"`，`verification.unsupported_claims` 写入错误码；
  由 Composer 输出"无法生成安全可执行的计划"。Phase 2 MVP 不降级到单 agent。

参见 `docs/phase2-workflow-agent.md` §5。
"""

from __future__ import annotations

import uuid
from typing import Any

from app.constants.model_profiles import PROFILE_ROUTING
from app.constants.workflow import (
    ERR_PLAN_LLM_FAILED,
    ERR_PLAN_UNKNOWN_TOOL,
    NODE_PLANNER,
    STEP_AGENT_TOOL,
    WORKFLOW_STATUS_FAILED,
    WORKFLOW_STATUS_PENDING,
)
import sys

from app.llm.retry import LLMCallError
from app.prompts.workflow import build_planner_system_prompt, build_planner_user_prompt
from app.state import AgentState
from app.utils.logger import log_node, log_warning
from app.workflow import (
    Plan,
    PlanValidationError,
    parse_plan,
)
from app.workflow.registry import ToolRegistry, default_tool_registry


def _new_plan_id() -> str:
    """独立函数便于测试 monkeypatch。"""

    return uuid.uuid4().hex


def _call_planner_llm(system_prompt: str, user_prompt: str) -> str:
    """调底层 chat completion 取 text；失败抛 LLMCallError。

    通过 `sys.modules["app.llm.chat"]` 动态解引用：`app.llm` 在 `__init__.py` 里把
    `chat` 函数 re-export 成了包属性，直接 `import app.llm.chat` 会拿到函数而非模块。
    这里走 sys.modules 才能让 conftest 里 `monkeypatch.setattr(llm_chat_mod,
    "_create_chat_completion", ...)` 对 planner 生效。
    """

    llm_chat_mod = sys.modules["app.llm.chat"]
    res = llm_chat_mod._create_chat_completion(
        profile=PROFILE_ROUTING,
        trace_stage="planner",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (res.choices[0].message.content or "").strip()


def _validate_plan_tools(plan: Plan, registry: ToolRegistry) -> list[str]:
    """校验 plan 里 tool_agent step 引用的工具都在 registry；失败返回未知工具列表。

    `WorkflowStep` 层面已经保证了 tool_agent step 必须带 tool，这里只补"名字必须
    能映射到已注册 function name"。参数/越权等更细的检查留给 Verifier（Phase 2 PR-3）。
    """

    unknown: list[str] = []
    for step in plan.steps:
        if step.agent != STEP_AGENT_TOOL or not step.tool:
            continue
        if not registry.has(step.tool):
            unknown.append(step.tool)
    return unknown


def _fail(plan_id: str, code: str, detail: str = "") -> dict[str, Any]:
    """把一次 planner 失败翻译成 state 增量。

    - `plan` 置空：Executor 用 `plan.get("steps")` 判空即可短路；
    - `verification.unsupported_claims`：Composer 据此生成拒绝文案（PR-4 实现，
      MVP 先把错误码落进 state 供 trace 观察）；
    - `agent_outputs[planner]`：方便 LangSmith 里直接看到失败点。
    """

    return {
        "plan": {},
        "plan_id": plan_id,
        "workflow_status": WORKFLOW_STATUS_FAILED,
        "verification": {
            "status": "failed",
            "missing_fields": [],
            "unsupported_claims": [code],
            "risk_warnings": [],
        },
        "agent_outputs": {
            NODE_PLANNER: {
                "status": "failed",
                "error_code": code,
                "detail": detail,
            }
        },
        "debug_info": {
            NODE_PLANNER: {
                "plan_id": plan_id,
                "status": "failed",
                "error_code": code,
                "detail": detail,
            }
        },
    }


def planner_node(
    state: AgentState,
    *,
    registry: ToolRegistry | None = None,
) -> AgentState:
    """Planner 节点主入口。

    `registry` 参数仅用于测试注入裁剪后的 registry；生产走默认全局 registry。
    """

    registry = registry or default_tool_registry
    plan_id = _new_plan_id()
    message = state["messages"][-1]["content"].strip()
    auth = state["auth"]

    system_prompt = build_planner_system_prompt(registry.visible_tools(auth))
    user_prompt = build_planner_user_prompt(message, auth.role)

    try:
        raw = _call_planner_llm(system_prompt, user_prompt)
    except LLMCallError as exc:
        log_warning(
            NODE_PLANNER,
            "planner LLM failed; fail-closed",
            {
                "plan_id": plan_id,
                "code": exc.code,
                "profile": exc.profile,
                "provider": exc.provider,
                "model": exc.model,
            },
        )
        return _fail(plan_id, ERR_PLAN_LLM_FAILED, detail=exc.code)

    try:
        plan = parse_plan(raw)
    except PlanValidationError as exc:
        log_warning(
            NODE_PLANNER,
            "planner output failed schema validation; fail-closed",
            {
                "plan_id": plan_id,
                "error_code": exc.code,
                "response_preview": raw[:200],
            },
        )
        return _fail(plan_id, exc.code, detail=exc.message)

    unknown_tools = _validate_plan_tools(plan, registry)
    if unknown_tools:
        log_warning(
            NODE_PLANNER,
            "planner referenced unknown tools; fail-closed",
            {"plan_id": plan_id, "unknown_tools": unknown_tools},
        )
        return _fail(
            plan_id,
            ERR_PLAN_UNKNOWN_TOOL,
            detail=",".join(unknown_tools),
        )

    # 成功路径：把 plan 序列化成 dict（state 层约定 plan: dict）。
    plan_dict: dict[str, Any] = {
        "task_type": plan.task_type,
        "compose_goal": plan.compose_goal,
        "steps": [
            {
                "id": step.id,
                "agent": step.agent,
                "purpose": step.purpose,
                "tool": step.tool,
                "args": dict(step.args),
                "query": step.query,
                "depends_on": list(step.depends_on),
            }
            for step in plan.steps
        ],
    }

    log_state = {**state, "plan_id": plan_id, "plan": plan_dict}
    log_node(
        NODE_PLANNER, log_state, extra={"planId": plan_id, "taskType": plan.task_type}
    )

    return {
        "plan": plan_dict,
        "plan_id": plan_id,
        "workflow_status": WORKFLOW_STATUS_PENDING,
        "agent_outputs": {
            NODE_PLANNER: {
                "status": "ok",
                "task_type": plan.task_type,
                "step_count": len(plan.steps),
            }
        },
        "debug_info": {
            NODE_PLANNER: {
                "plan_id": plan_id,
                "status": "ok",
                "task_type": plan.task_type,
                "step_count": len(plan.steps),
                "compose_goal": plan.compose_goal,
            }
        },
    }


# 防止 lint 对未使用 import 报警；保持显式导出便于外部测试 hook。
__all__ = ["planner_node"]

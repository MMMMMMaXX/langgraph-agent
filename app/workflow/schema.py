"""Plan / WorkflowStep schema + JSON 解析。

Planner 输出约定 —— 为什么这样设计：
- **业务名 vs function name**：Planner prompt 只暴露业务名（带点号，如
  `monitor.query_errors`），代码侧真正调用 LLM function-calling 时用下划线 name。
  schema 层接受业务名（或等价的下划线 name，防 LLM 偶尔偷懒），由 Verifier/Executor
  通过 `ToolRegistry` 转换。
- **不接受 `requires_confirmation` 字段**：是否需要确认完全由 `ToolMetadata` 派生，
  避免 Planner 瞎关安全开关。若 LLM 仍输出该字段，schema 会悄悄丢弃并记录。
- **Composer 不是 step**：`StepAgent` 不含 `composer`。末端合成由独立 composer_node
  承担；Planner 如需表达最终交付目标，用 `Plan.compose_goal`。

失败处理：任何校验失败抛 `PlanValidationError(code, message)`。code 取自
`app/constants/workflow.py` 的 `ERR_PLAN_*`，由上层 planner_node 写入 verification。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from app.constants.workflow import (
    ERR_PLAN_ARGS_INVALID,
    ERR_PLAN_DAG_CYCLE,
    ERR_PLAN_SCHEMA_INVALID,
    ERR_PLAN_STEP_LIMIT,
    MAX_PLAN_STEPS,
    STEP_AGENTS,
    STEP_AGENT_CHAT,
    STEP_AGENT_RAG,
    STEP_AGENT_TOOL,
    STEP_ID_PREFIX,
)


class PlanValidationError(ValueError):
    """Plan 校验失败。`code` 直接写入 verification.unsupported_claims。"""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code


class WorkflowStep(BaseModel):
    """单个 step。

    - `tool`：存业务名（含点号）或等价下划线 name，Executor 再通过 registry 映射。
    - `args`：调用 tool 时的参数；由 Verifier 按 ToolMetadata.schema 校验完整性。
    - `query`：rag_agent step 使用。
    - `depends_on`：仅支持前向引用，MVP 不做并发。
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str = Field(..., min_length=2, max_length=16)
    agent: str
    purpose: str = Field(..., min_length=1, max_length=500)
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    query: str | None = None
    depends_on: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not v.startswith(STEP_ID_PREFIX):
            raise ValueError(f"step id must start with {STEP_ID_PREFIX!r}")
        suffix = v[len(STEP_ID_PREFIX) :]
        if not suffix.isdigit() or int(suffix) <= 0:
            raise ValueError("step id suffix must be positive integer")
        return v

    @field_validator("agent")
    @classmethod
    def _validate_agent(cls, v: str) -> str:
        if v not in STEP_AGENTS:
            raise ValueError(f"agent must be one of {STEP_AGENTS}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _validate_agent_fields(self) -> WorkflowStep:
        # tool_agent step 必须带 tool；rag_agent step 必须带 query；
        # 这里是结构约束，Verifier 再做业务级别的参数/越权检查。
        if self.agent == STEP_AGENT_TOOL and not self.tool:
            raise ValueError("tool_agent step requires tool")
        if self.agent == STEP_AGENT_RAG and not self.query:
            raise ValueError("rag_agent step requires query")
        if self.agent == STEP_AGENT_CHAT and (self.tool or self.query):
            # chat_agent 不该携带 tool / query，避免 Planner 把无处消费的字段塞进来。
            raise ValueError("chat_agent step must not set tool or query")
        return self


class Plan(BaseModel):
    """Planner 输出的顶层 plan。

    注意：DAG / 重复 id / step 上限这些"整体"约束不在 pydantic validator 里做，
    而是由 `parse_plan` 在构造完成后显式检查。原因：pydantic v2 把 validator 抛出的
    `ValueError` 子类（包括我们的 `PlanValidationError`）吞进 `ValidationError`，
    丢掉 `.code` 字段，没法精确区分"结构错 / DAG 错 / 超限错"。
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    task_type: str = Field(..., min_length=1, max_length=64)
    steps: tuple[WorkflowStep, ...] = Field(..., min_length=1)
    compose_goal: str = ""


def _check_plan_invariants(plan: Plan) -> None:
    """Plan 构造完成后的整体约束检查，直接抛带 code 的 PlanValidationError。"""

    if len(plan.steps) > MAX_PLAN_STEPS:
        raise PlanValidationError(
            ERR_PLAN_STEP_LIMIT,
            f"plan has {len(plan.steps)} steps, exceeds MAX_PLAN_STEPS={MAX_PLAN_STEPS}",
        )

    seen: set[str] = set()
    for step in plan.steps:
        if step.id in seen:
            raise PlanValidationError(
                ERR_PLAN_SCHEMA_INVALID,
                f"duplicated step id: {step.id}",
            )
        for dep in step.depends_on:
            if dep == step.id:
                raise PlanValidationError(
                    ERR_PLAN_DAG_CYCLE,
                    f"step {step.id} depends on itself",
                )
            if dep not in seen:
                raise PlanValidationError(
                    ERR_PLAN_DAG_CYCLE,
                    f"step {step.id} has forward/unknown dep {dep}",
                )
        seen.add(step.id)


def parse_plan(raw: str | dict) -> Plan:
    """把 LLM 输出（字符串或已解析 dict）校验为 Plan。

    失败一律抛 `PlanValidationError`，由 planner_node 翻译成 workflow_status=failed。
    之所以把 json 解析放在这里：LLM 偶尔会在 JSON 前后带一点自然语言，这里一次性
    做"第一个 `{` 到最后一个 `}` 之间的子串"兜底，减少 planner_node 的样板代码。
    """

    if isinstance(raw, dict):
        payload: Any = raw
    else:
        text = raw.strip()
        if not text:
            raise PlanValidationError(ERR_PLAN_SCHEMA_INVALID, "empty plan payload")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # 尝试截取第一个 JSON 对象：LLM 偶尔会在外层包一些 ```json``` 或解释语。
            first, last = text.find("{"), text.rfind("}")
            if first == -1 or last == -1 or last <= first:
                raise PlanValidationError(
                    ERR_PLAN_SCHEMA_INVALID, "plan payload is not valid JSON"
                ) from None
            try:
                payload = json.loads(text[first : last + 1])
            except json.JSONDecodeError as exc:
                raise PlanValidationError(
                    ERR_PLAN_SCHEMA_INVALID, f"plan JSON decode error: {exc}"
                ) from None

    if not isinstance(payload, dict):
        raise PlanValidationError(
            ERR_PLAN_SCHEMA_INVALID, "plan payload must be object"
        )

    # 主动剔除 LLM 可能塞进来但不允许 Planner 声明的字段。
    # requires_confirmation 由 ToolMetadata 派生，Planner 不得自行开关。
    payload.pop("requires_confirmation", None)

    try:
        plan = Plan.model_validate(payload)
    except ValidationError as exc:
        # 精准区分参数型错误 vs 一般结构错误。
        code = ERR_PLAN_SCHEMA_INVALID
        for err in exc.errors():
            loc = err.get("loc", ())
            if "args" in loc:
                code = ERR_PLAN_ARGS_INVALID
                break
        raise PlanValidationError(code, str(exc)) from None

    _check_plan_invariants(plan)
    return plan


__all__ = [
    "Plan",
    "PlanValidationError",
    "WorkflowStep",
    "parse_plan",
]

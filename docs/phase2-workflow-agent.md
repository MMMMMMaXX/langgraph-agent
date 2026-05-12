# Phase 2 设计草稿：Workflow Agent MVP

对应总体设计：`docs/enterprise-multi-agent-design.md` §13 Phase 2、§4.2/4.5/4.7、§8。
依赖：`docs/phase1-auth-tool-safety.md`（AuthContext、ToolMetadata、Confirmation、Idempotency、ExecutionRecord 已就绪）。

本草稿只覆盖 Workflow MVP 所需的最小闭环，**不含**异步 `POST /workflow` + worker、
跨 session 长耗时任务、Multi-hop RAG（Phase 3）、MCP adapter（Phase 4）。目标是在
**单次同步请求**里跑通 `Planner → 分派执行 → Verifier → Composer` 的 DAG 编排，
并 100% 复用 Phase 1 的工具安全闭环。

> 说明：Phase 2 MVP **不承诺跨请求 workflow 恢复**。当前 `session_runtime` 每轮
> 会重置 `pending_confirmation / tool_executions`（见 `app/runtime/session_runtime.py:92`），
> `plan / step_results` 也不持久化。二次请求带 token 走的仍是 Phase 1 单 step 的
> 确定性重放（Scheme A），workflow 本身按"重新规划 + 命中 idempotency key 自动去重"
> 处理。真正的"只复跑某 step"能力依赖 LangGraph checkpoint 持久化 `plan/step_results`，
> 推迟到 **Phase 2.5**。

### 术语约定：业务工具名 vs function name

OpenAI function-calling 要求 `name` 匹配 `^[a-zA-Z0-9_-]+$`，不能含点号。文中 plan /
prompt / 设计文档一律使用 **业务工具名**（点号风格，语义清晰）；代码侧 function
calling 走 **function name**（下划线风格，见 `app/constants/tooling.py`）。

| 业务工具名（plan / doc） | function name（代码）                                    |
| ------------------------ | -------------------------------------------------------- |
| `monitor.query_errors`   | `monitor_query_errors`                                   |
| `ticket.create`          | `ticket_create`（已存在，`app/constants/tooling.py:12`） |
| `knowledge.search`       | `knowledge_search`                                       |

`app/workflow/registry.py` 负责双向映射；Planner prompt 只暴露业务名，避免 LLM
瞎编下划线变体。

---

## 1. 范围

### 必做（Phase 2 卡点）

1. **Planner Agent**：LLM + JSON schema，把 user message 拆成结构化 plan
   （`task_type` + `steps[]`），每个 step 绑定 `agent` 与 `purpose`。
2. **Workflow State**：AgentState 新增 `plan / step_results / workflow_status`，
   支持单次请求内分步执行与中断短路。**跨请求恢复不在 MVP**（见顶部说明）。
3. **Workflow Executor**：顺序/受限 DAG 驱动 step 逐个执行，可 short-circuit 到
   `need_confirmation` 或 `failed`。**不直接复用** `tool_agent_node / rag_agent_node`
   （它们读最后一条 user message，见 `app/agents/tool_agent.py:111`），而是抽出
   `run_tool_step() / run_rag_step()` 辅助函数，按 step 指定的 `tool / args / query`
   执行，避免主消息覆盖 step 输入。
4. **Verifier Agent**：检查 plan 输出的 evidence / tool args / 越权 / 风险等级，
   输出 `pass | need_clarification | need_confirmation | failed`。
   **MVP 纯规则驱动**；LLM claim-level judge 默认关（与总设 §10.4 一致，仅离线 eval 或显式开关）。
5. **Response Composer**：统一拼装 `answer + completed_actions + pending_confirmations
   - missing_information + citations`，替换当前 `merge_node` 的窄路径。
6. **Supervisor 扩展**：新增常量 `ROUTE_WORKFLOW = "workflow_agent"` 与
   `intent="workflow"`；`AgentState.routes` 的 Literal 同步扩展（见 `app/state.py:48`、
   `app/graph.py:27` 路由表），避免在文档里留下 "workflow" / "planner" 二选一的模糊。
7. **Mock Enterprise Tools**：新增 `monitor.query_errors`（read_only）。
   **不新增** `ticket.create_draft`：生成草稿文本属于 Composer 职责（不进工具层）；
   若业务真需要落草稿表，它本质是副作用工具，必须走 Phase 1 confirmation / idempotency
   闭环，而不是伪装成 read_only。真正副作用沿用已有 `ticket_create`（`app/tools/ticket.py:64`）。
8. **Workflow Eval**：新增 `workflow_*` case，覆盖
   - 单步 read_only plan
   - 多步 read_only（监控 → 知识库 → Composer 生成草稿文本）
   - plan schema 违规被拒
   - step 失败 → `partial` 状态
   - side_effect step → `need_confirmation` → 二次 token 请求命中 Phase 1 闭环

### 不做（明确推迟）

- 异步 `POST /workflow` + background worker → Phase 2.5 / Phase 7
- 跨 session workflow 持久化（workflow_id 可复用） → Phase 2.5
- 真实 MCP tool → Phase 4
- Multi-hop RAG query decomposition → Phase 3
- 文档 ACL、高敏不可见 → Phase 5

---

## 2. 数据结构变更

### 2.1 AgentState 新增字段

在 `app/state.py:AgentState` 追加：

```python
# Phase 2：Planner 输出的结构化 plan，Verifier/Executor 消费。
# 结构：{"task_type": str, "steps": [WorkflowStep], "compose_goal": str}
# 注意：不含 requires_confirmation；是否需要确认完全由 ToolMetadata 派生，
# Verifier 阶段逐 step 查 registry，防 Planner 瞎关安全开关。
plan: dict
plan_id: str  # uuid4，Planner 生成，便于 trace/debug/eval 聚合

# Phase 2：每个 step 执行完的产物，key = step.id
# 内部状态对齐总设 §8.1：
#   pending / running / waiting_user / need_confirmation / succeeded / failed / skipped
step_results: Annotated[dict, merge_dict]

# Phase 2：整个 workflow 的内部终态（对齐总设 §8.1）：
#   pending | running | waiting_user | need_confirmation | succeeded | failed | partial
# Composer 再把 succeeded 对外映射成 "completed" 文案；内部/代码/trace 只用内部枚举，
# 避免枚举漂移。
workflow_status: str

# Phase 2：Verifier 的结构化输出，Composer 消费（risk_warnings 会进 answer）
verification: dict
```

`routes` 的 Literal 追加常量 `ROUTE_WORKFLOW = "workflow_agent"`，同步更新
`app/state.py:48` 与 `app/graph.py:27` 的路由表；不再使用 `"workflow" / "planner"`
二义写法。

### 2.2 WorkflowStep schema

新建 `app/workflow/schema.py`：

```python
StepAgent = Literal["tool_agent", "rag_agent", "chat_agent"]
# 注意：不含 "composer"。Composer 是 workflow 末端独立节点，不作为 plan 里的 step，
# 避免"plan 里声明一个 composer step + 末端再跑一次 Composer"的双执行歧义。
# Planner 如需表达"最后要合成什么"，用 plan 顶层字段 compose_goal/final_deliverable
# 而不是 step。

@dataclass(frozen=True)
class WorkflowStep:
    id: str                                  # "s1","s2"...
    agent: StepAgent
    purpose: str
    tool: str | None = None                  # 当 agent=tool_agent 时必填（业务名，registry 映射到 function name）
    args: dict[str, Any] = field(default_factory=dict)  # tool 调用参数；由 Planner 产出、Verifier 校验
    query: str | None = None                 # 当 agent=rag_agent 时必填
    depends_on: tuple[str, ...] = ()         # 仅支持前向引用，MVP 不做并发

@dataclass(frozen=True)
class Plan:
    task_type: str
    steps: tuple[WorkflowStep, ...]
    compose_goal: str = ""                   # 给 Composer 的最终合成目标，非 step
    # requires_confirmation 不再由 Planner 显式输出：
    # 由 ToolMetadata.requires_confirmation 在 Verifier 阶段按 step.tool 派生，
    # 防 Planner 瞎关安全开关。
```

Planner 输出必须通过 pydantic/jsonschema 校验，超过 `MAX_STEPS`（建议 6）直接拒。

---

## 3. 模块拆分

```
app/
  agents/
    planner_agent.py        # 新增：Planner LLM 调用 + schema 校验
    verifier_agent.py       # 新增：Verifier 检查
    composer_agent.py       # 新增：替换 merge_node 的最终答复拼装
    workflow_executor.py    # 新增：DAG/顺序驱动
  workflow/
    schema.py               # WorkflowStep / Plan pydantic model
    registry.py             # task_type → 允许使用的 agent/tool 白名单
  constants/
    workflow.py             # MAX_STEPS, 状态枚举, Planner prompt 常量
  prompts/
    workflow.py             # Planner/Verifier/Composer prompt
  tools/
    monitor.py              # 新增 read_only mock tool: monitor_query_errors
    # ticket.py 已有 ticket_create（side_effect），Phase 2 不新增草稿工具
```

---

## 4. 执行流

```
API (/chat)
  │
  ▼
supervisor_node
  │ confirmation_token 存在 → routes=[ROUTE_TOOL_AGENT]（直接走 Phase 1 Scheme A 重放，
  │                            不进 Planner，避免重新规划导致 args 变化 / idempotency key 漂移）
  │ 否则 intent=workflow → routes=[ROUTE_WORKFLOW]
  ▼
planner_node          # LLM → plan JSON → schema 校验（含 plan_id = uuid4）
  │ plan 不合法 → workflow_status=failed，直接跳 composer（fail-closed）
  ▼
workflow_executor     # 按 depends_on 顺序，逐 step 分派
  │   step.agent = tool_agent → run_tool_step(step, auth, state)  # 不调 tool_agent_node
  │   step.agent = rag_agent  → run_rag_step(step, auth, state)   # 不调 rag_agent_node
  │   step.agent = chat_agent → run_chat_step(step, state)
  │ 某步 need_confirmation → 立即短路，pending_confirmation 冒泡到 state，跳 composer
  │ 某步 failed → workflow_status=partial，跳 composer
  ▼
verifier_node         # 检查 step_results 证据/越权/缺参；按 ToolMetadata 派生 requires_confirmation
  │ verification.status=need_confirmation → 更新 workflow_status
  ▼
composer_node         # 末端独立节点（不作为 plan step），合成 answer + completed_actions
  ▼                     + pending_confirmations，按 plan.compose_goal 指导文本
memory_node → END
```

LangGraph 图变化：

- `graph.py` 增加 `NODE_PLANNER / NODE_EXECUTOR / NODE_VERIFIER / NODE_COMPOSER`。
- Supervisor 分支追加 `ROUTE_WORKFLOW`；non-workflow 请求仍走老链路，让 composer
  兼容 `plan` 为空的情况（MVP 选择：不改 merge_node 命名，只新增 composer_node，
  在 workflow 支路替换；后续 PR 再统一），降低回归风险。

**复用策略说明**：为什么不直接调 `tool_agent_node / rag_agent_node`？
它们读 `state["messages"][-1]`（见 `app/agents/tool_agent.py:111`），而 workflow
step 的输入来自 `step.tool / step.args / step.query`，主消息可能早已与 step 无关。
所以把 tool 执行、rag 检索的"纯函数"部分抽成 `run_tool_step() / run_rag_step()`，
tool*agent_node / rag_agent_node 退化成"消息入口 → step 构造 → run*\*\_step"的
薄适配层，保证两条路径共享同一份副作用安全闭环。

---

## 5. Planner 设计

### 5.1 Prompt 结构

```
System:
  你是企业排障/协作任务的 Planner。只输出 JSON，不输出散文。
  可用 agent：tool_agent / rag_agent / chat_agent（不含 composer，末端合成由系统处理）
  可用 tool：{tool_registry 过滤后按 auth.role 白名单给出（使用业务名，如 monitor.query_errors）}
  规则：
  - 最多 {MAX_STEPS} 步
  - 步骤 id 从 s1 递增，depends_on 只能引用前序
  - step.tool 必须带完整 args（键名与 ToolMetadata.schema 一致）
  - 不要输出 requires_confirmation 字段；是否需要确认由系统按 ToolMetadata 派生
  - 顶层可选 compose_goal 描述最终要交付什么（给 Composer 指令，不是 step）

User:
  {user_message}
  auth.role={role}   # Planner 据此裁剪可用 tool
```

### 5.2 输出校验

- pydantic `Plan` model 校验字段、枚举、长度。
- 额外语义校验：
  - `step.tool` 业务名必须能映射到 registry 内的 function name，且 `auth.role` 可访问
  - `step.args` 必须通过 ToolMetadata.schema 校验（缺参/越界直接拒）
  - `requires_confirmation` 由 Verifier 按 ToolMetadata 派生（Planner 不声明；
    如果 Planner 输出了该字段，直接忽略并记 `unsupported_claims`）
  - DAG 无环

校验失败 → `workflow_status=failed`，`verification.unsupported_claims=["plan_schema_invalid"]`，
**一律 fail-closed 由 Composer 输出"无法生成安全可执行的计划"**，不 fallback 到
单 agent（避免 Planner 拒的请求被 chat_agent 绕过安全检查）。

---

## 6. Verifier 设计

**MVP 纯规则驱动**：在线链路不跑 LLM claim judge（与总设 §10.4 一致，claim-level
verification 默认不进在线链路，仅离线 eval 或显式开关开启）。

| 检查项                                      | 实现                                                    |
| ------------------------------------------- | ------------------------------------------------------- |
| 每个 side_effect step 是否经过 confirmation | 读 step_results[*].output.status                        |
| tool 参数是否完整                           | 读 execution_record                                     |
| 是否越权                                    | auth.role vs tool registry                              |
| risk_level 是否 >=medium                    | tool metadata 聚合                                      |
| 是否有未支持的断言（claim-level）           | **默认关**，仅 `WORKFLOW_LLM_VERIFIER=true` 或离线 eval |

输出：

```json
{
  "status": "pass | need_clarification | need_confirmation | failed",
  "missing_fields": [],
  "unsupported_claims": [],
  "risk_warnings": ["side_effect_requires_user_confirmation"]
}
```

`verification.status` 与 `workflow_status` 通过优先级合并：
`failed > need_confirmation > need_clarification > partial > succeeded`
（Composer 对外把 `succeeded` 映射为 "completed" 文案）。

---

## 7. Composer 设计

输入聚合：

- `plan.steps` + `step_results` → completed_actions
- `pending_confirmation`（若有）→ pending_confirmations
- `verification.risk_warnings` → 附加到 answer 末尾
- `rag_agent` step 产出的 citations → citations

输出 schema（写回 `answer` + 结构化字段放 `agent_outputs["composer"]`）：

```json
{
  "answer": "...",
  "completed_actions": [{"step": "s1", "tool": "monitor.query_errors", "summary": "..."}],
  "pending_confirmations": [{"tool": "ticket.create", "token": "...", "expires_at": "..."}],
  "missing_information": [],
  "citations": [...]
}
```

Composer **不发起**新的副作用，只做文字合成。这保证 workflow 的副作用入口永远是 tool_agent，Phase 1 的安全闭环不会被绕过。

---

## 8. 与 Phase 1 的衔接

- **AuthContext**：Planner prompt、Verifier 白名单、tool_agent 执行层都已透传，无需改造。
- **Confirmation Token**：某 step 触发 `need_confirmation` 时，tool_agent（或
  `run_tool_step`）产出的 pending_confirmation 直接冒泡到 state。客户端二次请求
  带 token 走 Phase 1 Scheme A 单 step 直解，**MVP 不做跨请求 workflow 续跑**；
  客户端二次请求会重新规划整条 plan，已 side_effect 的 step 靠 idempotency key
  自动去重（`tool_agent` 返回已有 execution）。
- **Idempotency**：workflow 复跑命中 idempotency key → `tool_agent` 返回已有
  execution，`step_results[*].output.status="deduplicated"`，Composer 对外表达为
  "该操作已在此前完成"。
- **Execution Record**：`workflow_id / step_id` **不进表**（SQLite 不支持
  `ALTER TABLE ... IF NOT EXISTS`，避免为一个 trace 字段引入迁移脚本）；作为
  `debug_info / trace` 字段透出给 LangSmith/日志即可。真正落表推迟到 Phase 2.5
  随 checkpoint 方案一起做一次轻量 schema migration。

---

## 9. PR 拆分

1. **PR-1 Plan schema + Planner node + Supervisor workflow intent**
   `app/workflow/schema.py / registry.py`、`app/agents/planner_agent.py`、prompts；
   Supervisor 识别多步排障 → `routes=[ROUTE_WORKFLOW]`，让 Planner 有入口。
2. **PR-2 Workflow Executor + AgentState 字段 + `run_tool_step / run_rag_step` 抽取**
   `workflow_executor.py`、`app/state.py` 扩字段、`graph.py` 增节点；
   把 tool_agent_node / rag_agent_node 的执行核心抽出成可复用 helper。
   覆盖：顺序执行、失败短路、confirmation 短路。
3. **PR-3 Verifier（纯规则）**
   `verifier_agent.py`；verification 聚合到 workflow_status（对齐总设 §8.1 枚举）。
4. **PR-4 Composer 替换 merge**
   新 `composer_agent.py`；`merge_node` 保留薄 shim（非 workflow 请求直接透传），
   后续 PR 再统一收敛。
5. **PR-5 Mock tools + Workflow Eval**
   `tools/monitor.py`（read_only）；`scripts/eval_cases.json` 新增 5 个 workflow case；
   `eval_chat.py` 新增 `workflow_success_rate / plan_schema_pass_rate /
confirmation_bridge_rate` 三项指标。

---

## 10. 验收 / DoD

- [ ] 现有全量单测保持绿；新增 workflow 相关单测覆盖率 ≥ 80%。
- [ ] `scripts/eval_chat.py` workflow 子集全过；tool_safety 旧 case 不回归。
- [ ] Demo：`"帮我分析 payment-service 最近 30 分钟 5xx 增加的原因，并给一个排查工单草稿"`
      → 规划 2 步（`monitor.query_errors` → `knowledge.search`）+ compose_goal="排查工单草稿"
      → Composer 合成草稿文本，全 read_only 无需确认即闭环。
- [ ] Demo 升级版：把 compose_goal 改为真正建单 → Planner 追加 `ticket.create`
      （side_effect）step → 第一次返回 `need_confirmation`；第二次请求带
      `confirmation_token` 时 Supervisor **直接路由到 tool_agent** 走 Phase 1
      Scheme A 重放，不再进 Planner，保证 args 不变 / idempotency key 稳定。
- [ ] Trace / debug_info 可以看到 `plan_id / step_id / workflow_status`
      （`plan_id` 由 Planner 生成进 AgentState，非 workflow 请求该字段为空）。

---

## 11. 风险与预案

| 风险                                      | 预案                                                                                                                                                                      |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Planner 产出不稳定                        | JSON schema 强校验 + 低温度；**仅 low-risk / 全 read_only** 请求可 fallback 到单 agent；一旦 plan 涉及 side_effect 或越权 → fail-closed 返回 `need_clarification`，不降级 |
| Workflow 改动冲击现有 chat/rag 回归       | 新路径通过 `intent=workflow` 独立分支；merge_node 作为非 workflow 请求的兼容 shim                                                                                         |
| Executor 变成"小 LangGraph" 复杂度爆炸    | MVP 只支持顺序 + 前向 `depends_on`，不做并发/循环                                                                                                                         |
| Verifier 误杀正常请求                     | MVP 纯规则；LLM claim judge 仅离线 eval 或显式开关；fail-open 仅限 low risk                                                                                               |
| 跨请求 workflow 续跑语义复杂              | MVP 明确不做：二次请求重新规划 + 靠 idempotency 去重；checkpoint 持久化推迟到 Phase 2.5                                                                                   |
| 二次 confirmation 请求被重新规划打断      | Supervisor 在 `confirmation_token` 存在时**硬绑**到 `ROUTE_TOOL_AGENT`，直接走 Phase 1 Scheme A 重放，不进 Planner；避免 LLM 重规划导致 args / idempotency key 漂移       |
| tool/rag 节点被 workflow 复用时输入被覆盖 | 抽 `run_tool_step / run_rag_step` 纯函数；tool_agent_node 保留为消息入口适配层，两条路径共享                                                                              |

---

## 12. 面试 / 简历表达

- "在 Phase 1 把副作用工具的 Auth + Confirmation + Idempotency + ExecutionRecord
  做成了可复用的安全闭环之后，Phase 2 把单 agent 升级为 Planner → Executor →
  Verifier → Composer 的多 agent 编排，支持结构化 plan 驱动的多步任务。"
- "Planner 用 JSON schema 硬约束输出，Verifier 用规则 + 白名单防越权，Composer
  只做合成不碰副作用——安全入口收敛在 tool_agent 一处，Phase 1 的闭环不会被绕过。"
- "跨请求行为：MVP 不持久化 workflow，二次请求重新规划；已 side_effect 的 step
  靠 Phase 1 idempotency key 命中 tool_executions 主键抢占，直接返回历史结果，
  不会重复扣款式地再执行。真正的 workflow 级断点续跑（持久化 plan/step_results）
  留给 Phase 2.5，依赖 LangGraph checkpoint。"

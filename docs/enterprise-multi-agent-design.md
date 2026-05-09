# 企业知识工作流 Multi-Agent Assistant 设计文档

## 1. 背景与目标

当前项目已经具备企业知识库问答、文档导入、混合检索、引用、Tool Calling、
会话记忆、LangSmith tracing、Docker 部署和 eval harness 等基础能力。下一阶段
目标不是继续堆单点能力，而是把系统升级为一个面向企业内部知识和业务流程的
Multi-Agent Assistant。

这个系统的定位是：

> 面向企业内部知识、权限、工具和流程的多智能体协作平台，支持从内部知识库
> 查证据、从业务系统取数据、执行低风险工具、生成可追溯方案，并通过评测和
> tracing 持续优化。

它与个人助手、Codex、Claude Code 的区别在于：个人助手更关注单个用户即时任务，
而企业 Multi-Agent Assistant 更关注组织知识治理、权限边界、业务系统联动、
流程编排、可观测性和可审计性。

## 2. 核心业务场景

### 2.1 企业知识问答

用户通过如流机器人或 Web 前端提问：

```text
WAI-ARIA 技术是什么？
如何使用公司内部发布流程？
某个接口错误码 10021 代表什么？
```

系统从企业知识库 API 拉取 Markdown、JSON、HTML 等文档，经过解析、切片、
索引和检索后，返回带引用的答案。

### 2.2 研发/运维排障助手

用户提问：

```text
帮我分析 payment-service 最近 30 分钟 5xx 增加的原因，并给一个排查工单草稿。
```

系统需要：

1. 查询监控或日志工具。
2. 检索排障手册。
3. 汇总异常类型、影响范围、可能原因。
4. 生成排查步骤。
5. 生成工单草稿，等待用户确认后再创建。

### 2.3 跨系统方案生成

用户提问：

```text
基于 A 项目的接口文档和 B 项目的部署逻辑，写一个跨系统集成测试方案。
```

系统需要做 multi-hop query decomposition，分别检索 A 项目接口文档、B 项目部署
文档和联调约束，再综合生成测试方案。

### 2.4 跨部门知识联动

A 部门使用 Chroma + SQLite，B 部门使用 Elasticsearch + Milvus，且数据不能物理
合并。系统通过 Federated Agent Gateway 统一查询入口、权限上下文、结果融合和
引用格式。

## 3. 总体架构

```text
User / 如流 / Web
        |
        v
FastAPI Gateway
        |
        v
Session Runtime + Auth Context
        |
        v
LangGraph Orchestrator
        |
        +-- Supervisor / Router Agent
        +-- Planner Agent
        +-- Knowledge Agent
        +-- Tool Agent
        +-- Verification Agent
        +-- Memory Agent
        +-- Response Composer
        |
        v
Final Answer + Citations + Debug + Trace
```

外部依赖：

```text
Internal Knowledge API
Business Tools / MCP Servers
Monitoring / Ticket / CI Systems
SQLite Catalog
SQLite Conversation History
LangGraph SQLite Checkpointer
Chroma / Vector Store
LangSmith
Redis / Queue, future
```

## 4. Agent 职责划分

### 4.1 Supervisor / Router Agent

负责入口路由，不直接回答业务问题。

输入：

```text
user message
session summary
recent messages
auth context
```

输出：

```json
{
  "intent": "knowledge_qa | workflow | tool | chat | multi_hop",
  "routes": ["knowledge_agent", "tool_agent"],
  "risk_level": "low | medium | high",
  "reason": "..."
}
```

设计原则：

- 规则优先，LLM 兜底。
- 只做一次主路由，避免 Agent 之间互相踢皮球。
- routes 去重。
- 对高风险请求标记 `risk_level`，交给 Verification Agent 处理。

### 4.2 Planner Agent

负责复杂任务拆解，输出结构化 plan。

适用场景：

- 多步排障。
- 跨项目方案生成。
- 查知识库 + 调工具 + 生成草稿。
- multi-hop RAG。

输出示例：

```json
{
  "task_type": "incident_triage",
  "steps": [
    {
      "id": "s1",
      "agent": "tool_agent",
      "tool": "monitor.query_errors",
      "purpose": "查询最近 30 分钟 5xx 错误"
    },
    {
      "id": "s2",
      "agent": "knowledge_agent",
      "query": "payment-service 5xx 排障手册"
    },
    {
      "id": "s3",
      "agent": "composer",
      "purpose": "生成排查方案和工单草稿"
    }
  ],
  "requires_confirmation": false
}
```

约束：

- plan 必须通过 schema 校验。
- 最大 step 数受限。
- 不允许 Planner 直接执行副作用工具。

### 4.3 Knowledge Agent

负责企业知识库检索回答。

当前已有能力：

- 文档导入。
- SQLite catalog。
- SQLite FTS5 lexical retrieval。
- Chroma dense retrieval。
- hybrid merge。
- query classification。
- rerank。
- chunk merge。
- source-aware answer。
- citation。
- search inspect。
- knowledge health。

升级方向：

- multi-hop query decomposition。
- evidence group。
- source diversity。
- claim-level citation faithfulness verifier。
- ACL/DLS 前置过滤。
- federated retrieval adapter。

### 4.4 Tool Agent

负责调用内部工具或 MCP tools。

工具分类：

```text
read_only:
  查询监控、查询工单、查询项目状态

side_effect:
  创建工单、发消息、触发流程、修改配置
```

工具 metadata：

```json
{
  "name": "ticket.create",
  "read_only": false,
  "side_effect": true,
  "requires_confirmation": true,
  "idempotency_required": true,
  "risk_level": "medium"
}
```

副作用工具必须具备：

- confirmation flow。
- idempotency key。
- execution record。
- timeout/retry policy。
- audit log。

### 4.5 Verification Agent

负责检查计划和执行结果。

检查维度：

- 是否有足够 evidence。
- citation 是否合法。
- 是否存在 unsupported claim。
- 工具参数是否完整。
- 是否需要用户确认。
- 是否存在越权风险。
- 是否应该 fallback。

输出示例：

```json
{
  "status": "pass | need_clarification | need_confirmation | failed",
  "missing_fields": [],
  "unsupported_claims": [],
  "risk_warnings": []
}
```

### 4.6 Memory Agent / Memory Node

负责统一更新记忆，不允许各 Agent 随意写入长期记忆。

当前记忆分层：

```text
Session State:
  短期热状态，保存当前会话恢复所需的 messages + summary。
  存在 session cache 和 LangGraph checkpoint。

Summary:
  session 级压缩摘要，属于 AgentState.summary。

Conversation History:
  长期顺序事件流水，存在 SQLite history backend。
  用于审计、历史回顾和离线分析，不直接作为每轮实时上下文主来源。

Vector Memory:
  长期语义记忆，存在 Chroma memory collection。
```

边界说明：

```text
Session State:
  服务当前会话 resume 和短期上下文连续性。
  它可以包含最近 messages，但不是长期审计源。

Conversation History:
  服务长期历史查询和审计。
  实时对话优先使用 Session State；只有 summary/history 类问题才显式读取 history。

Vector Memory:
  服务长期语义召回。
  不保存完整对话流水，也不替代 history。
```

当前实现会同时写 Session State 和 Conversation History，但用途不同：

- Session State 是运行态快照，允许被 checkpoint 覆盖更新。
- Conversation History 是 append-only 事件记录，保留对话发生过的事实。
- 后续如果要减少重复存储，可以从 Conversation History 重建 Session State，但在线
  请求仍应优先读热状态或 checkpoint，避免每轮扫描历史事件。

写入策略：

- RAG 文档命中类定义问答：写 history，不写 vector memory。
- meta query：不写 vector memory，通常不写 history。
- bad answer / 资料不足：不写 vector memory。
- 用户偏好、稳定事实、项目上下文：按策略写 vector memory。

### 4.7 Response Composer

负责最终答复合成。

输入：

```text
agent outputs
tool results
knowledge evidence
verification result
memory context
```

输出：

```text
answer
citations
completed actions
pending confirmations
missing information
debug summary
```

## 5. 数据与存储设计

### 5.1 Knowledge Catalog

SQLite 是知识库真相源。

保存：

```text
documents
document_chunks
FTS5 lexical index
content_hash
parser_name
parser_version
metadata
```

Chroma 是可重建 dense index。

保存：

```text
id = chunk_id
document = chunk content
metadata = doc_id, title, source, chunk_index, security fields
embedding
```

两者通过 `doc_id` 和 `chunk_id` 对齐。

### 5.2 Session State

两层存储：

```text
session cache:
  进程内热缓存，保存 messages + summary。

LangGraph SQLite checkpoint:
  持久化 thread state，thread_id = session_id。
```

读取策略：

```text
cache 有有效 messages/summary -> 用 cache
否则 -> 从 checkpoint 恢复
否则 -> empty initial state
```

### 5.3 Conversation History

保存顺序问答事件：

```text
session_id
question
rewritten_query
answer
routes
tags
stored_to_vector
skip_reason
timestamp
```

用于：

- 总结所有问题。
- 历史审计。
- eval 隔离。

### 5.4 Vector Memory

保存长期语义记忆。

metadata：

```text
session_id
user_id, future
tenant_id, future
memory_type
tags
source_route
confidence
importance, future
created_at
updated_at
```

## 6. 权限与安全设计

### 6.1 文档级权限过滤

权限过滤必须前置到 retrieval，而不是 Top-k 后再过滤。

流程：

```text
request
-> auth context
-> retrieval filter
-> authorized candidates only
-> rerank
-> context build
-> answer
```

无权限内容不能进入：

- retrieval candidates。
- rerank prompt。
- LLM prompt。
- debug payload。
- LangSmith trace。
- citation list。

### 6.2 Prompt Injection via RAG

防护原则：

- 文档内容是 untrusted data，不是 instruction。
- 权限由系统层判断，不由 LLM 判断。
- 无授权 evidence 时拒答或资料不足。
- 高敏场景不暴露文档是否存在。

### 6.3 Tool 安全

副作用工具必须：

- requires confirmation。
- idempotency key。
- audit log。
- execution record。
- timeout 后状态 unknown，不盲目重试。

## 7. MCP 接入设计

MCP 用于标准化企业内部工具和数据源。

示例 MCP tools：

```text
knowledge.search
knowledge.fetch_doc
monitor.query_errors
ticket.search
ticket.create_draft
ticket.create
repo.search_code
ci.query_pipeline
```

MCP 返回大结果时，不直接塞给 LLM。

流程：

```text
tool raw result
-> result adapter
-> filtering / aggregation / feature extraction
-> compact evidence
-> LLM context
```

例如监控日志 10MB JSON：

```text
raw logs
-> group by error type
-> top stack traces
-> affected services
-> representative samples
-> compact evidence
```

## 8. Workflow Execution 设计

### 8.1 状态机

复杂任务应该有状态：

```text
pending
running
waiting_user
need_confirmation
succeeded
failed
partial
```

### 8.2 执行记录

每个 workflow 保存：

```text
workflow_id
session_id
user_id
plan
steps
status
agent_outputs
tool_executions
citations
errors
created_at
updated_at
```

### 8.3 幂等性

副作用 tool 的幂等键：

```text
idempotency_key = hash(session_id + tool_name + normalized_args)
```

注意：`request_id` 每次请求都会变化，不能进入幂等键，否则用户重试会得到不同 key，
失去幂等效果。`request_id` 只用于 audit log、trace 和问题定位。

如果客户端或上游系统支持显式重试，应优先使用用户侧或业务侧传入的稳定 key：

```text
idempotency_key = user_provided_retry_key
```

否则使用系统生成的稳定业务键：

```text
idempotency_key = hash(
  tenant_id
  + user_id
  + session_id
  + tool_name
  + normalized_args
)
```

超时处理：

```text
timeout
-> mark unknown
-> query by idempotency_key
-> found result: return existing
-> not found: safe retry with same key
```

## 9. Federated Agent 设计

用于跨部门知识联动，数据不能物理合并。

```text
Federated Agent Gateway
  |
  +-- Dept A Adapter: Chroma + SQLite
  +-- Dept B Adapter: Elasticsearch + Milvus
```

中心层职责：

- 统一入口。
- 统一 auth context。
- query planning。
- 调用部门 adapter。
- RRF / source diversity 融合。
- answer synthesis。

部门层职责：

- 本地 ACL。
- 本地检索。
- 本地 rerank。
- 返回最小 authorized evidence。

中心层数据最小化原则：

- 不保存原始文档。
- 默认只接收裁剪后的 authorized evidence。
- trace/debug 中只记录 doc_id、chunk_id、score、引用元数据和短 preview。
- 对高敏 evidence，中心层只接收摘要或 opaque evidence handle。
- LLM prompt 只包含最终需要回答的最小证据包。
- 如果必须跨部门传递原文片段，必须带权限上下文、审计记录和 retention policy。

也就是说，Federated Gateway 虽然是统一入口，但不能变成新的数据集中泄漏面。

## 10. Eval Harness 设计

### 10.1 RAG Eval

已有指标：

```text
route accuracy
top_k_hit
filtered_hit
rerank_hit
merged_hit
citation_hit
citation_refs_valid
answer_has_citation
fallback_accuracy
latency
```

### 10.2 Workflow Eval

新增指标：

```text
plan_valid
expected_steps_hit
expected_tools_hit
tool_args_valid
requires_confirmation_hit
side_effect_not_executed_without_confirmation
verification_status_hit
final_answer_quality
```

### 10.3 Memory Eval

新增指标：

```text
should_write_memory
should_not_write_memory
memory_recall_hit
memory_contamination
summary_restore_accuracy
preference_update_accuracy
```

### 10.4 Faithfulness Eval

新增指标：

```text
claim_count
supported_claim_count
unsupported_claim_count
citation_faithfulness_rate
```

成本控制：

- claim-level verification 不进入默认在线链路。
- 默认作为离线 eval 或抽样质检任务执行。
- 只对高风险答案、低置信答案、用户负反馈答案或关键业务知识运行。
- verifier 优先考虑小模型、NLI 模型或批处理方式，避免每条答案触发 5-10 次强模型调用。
- 在线链路只保留轻量规则校验，例如引用编号合法、引用是否来自本次 evidence。

## 11. 可观测性设计

每次请求记录：

```text
request_id
session_id
user_id, future
routes
query_type
plan_steps
tool_calls
retrieval_debug
citations
node_timings
token_usage
errors
```

LangSmith tags / metadata：

```text
node
route
query_type
doc_used
memory_used
tool_called
requires_confirmation
rerank_skipped
citation_count
```

RCA 辅助逻辑：

```text
top_k_hit=false -> parser/index/recall issue
filtered_hit=false -> threshold/filter issue
rerank_hit=false -> rerank issue
merged_hit=false -> chunk merge issue
citation_hit=false -> citation construction issue
all hit but answer wrong -> generation/faithfulness issue
```

## 12. 性能与扩展

### 12.1 中小规模

当前适合：

```text
SQLite FTS5
Chroma
FastAPI
SQLite checkpoint/history
```

### 12.2 百万文档规模

升级方向：

```text
SQLite FTS5 -> Elasticsearch / OpenSearch
Chroma -> Milvus / Qdrant / pgvector cluster
local session cache -> Redis
sync import -> async job queue
single worker -> horizontal replicas
```

迁移策略：

1. 双写阶段

   - 新导入文档同时写旧 SQLite/Chroma 和新 OpenSearch/Milvus。
   - 保留旧链路作为 fallback。

2. 回填阶段

   - 后台 job 分批把历史文档迁移到新索引。
   - 每批记录 doc_count、chunk_count、hash 和失败列表。

3. 双读评估阶段

   - 同一批 eval 同时跑旧检索和新检索。
   - 对比 top_k_hit、citation_hit、latency、token、cost。

4. 分数校准阶段

   - SQLite FTS5 的 bm25 口径和 Elasticsearch BM25 口径不同。
   - Chroma、Milvus、Qdrant 的 distance/similarity 方向也可能不同。
   - hybrid alpha/beta、threshold、rerank_top_k 必须重新基于 eval 校准。

5. 灰度切流阶段

   - 按租户、部门、文档类型或流量百分比灰度。
   - 保留快速回滚开关。

6. 下线旧索引
   - 新链路稳定后，旧索引只保留一段时间用于回溯。

一致性策略：

- SQLite catalog 或后续 Postgres catalog 仍作为 truth source。
- 检索索引是可重建派生数据。
- 索引同步采用 outbox/job 机制记录每个 doc_id 的 indexing status。
- 查询侧可根据 index_version 判断是否允许使用某个索引结果。

### 12.3 缓存

可引入：

```text
Redis session cache
embedding cache
retrieval result cache
semantic answer cache
tool result cache
```

Semantic cache 必须绑定：

```text
query similarity
permission scope
citation doc/chunk hash
knowledge index version
TTL
```

### 12.4 成本与 Token 预算

企业场景下，成本是系统设计约束，不是事后优化。

每次请求应记录：

```text
prompt_tokens
completion_tokens
embedding_tokens
rerank_tokens
tool_result_bytes
model_profile
estimated_cost
```

在线预算建议：

```text
simple_qa:
  small / default model
  short context
  no rerank if high confidence

definition_rag:
  max 1-2 evidence blocks
  short answer strategy

comparison_rag:
  source diversity
  controlled context budget

workflow:
  planner small model if schema-stable
  final answer strong model only when necessary
```

Model routing 原则：

- Intent classification、tool args extraction、title/summary 可优先使用小模型。
- 高风险最终回答、复杂综合推理、引用敏感回答使用强模型。
- 任何小模型输出必须经过 schema validation 和 fallback。

### 12.5 错误恢复与降级

不同节点失败时需要明确 fail-open 或 fail-closed。

```text
Auth / ACL failure:
  fail-closed，不返回数据。

Knowledge retrieval failure:
  可降级到部分来源或返回资料不足。

Tool read failure:
  可返回 partial answer，并说明哪个工具失败。

Tool side-effect unknown:
  不重复执行，进入 unknown/reconcile 状态。

Verifier failure:
  高风险任务 fail-closed。
  低风险总结类任务可 fail-open，但必须标记 confidence 降低。

LLM answer failure:
  返回可理解错误，并保留 request_id。
```

Response Composer 需要显式表达：

```text
completed
partial
need_clarification
need_confirmation
failed
```

### 12.6 用户反馈闭环

企业 Agent 需要从用户反馈中持续改进。

反馈入口：

```text
thumbs_up
thumbs_down
wrong_citation
missing_doc
bad_tool_call
answer_correction
```

反馈处理：

1. 记录 request_id、query、answer、citations、retrieval_debug、user feedback。
2. 负反馈进入 review queue。
3. 人工或半自动标注 root cause。
4. 生成新的 eval case。
5. 回归验证后再调整 chunking、retrieval、prompt 或 tool policy。

反馈不应直接写入长期记忆或知识库，必须经过审核或明确的数据治理流程。

### 12.7 异步与长耗时任务

部分工具不是同步返回，例如 CI trigger、长日志分析、大文档导入。

这些任务应进入异步 workflow：

```text
POST /workflow
-> return workflow_id
-> background worker executes steps
-> GET /workflow/{id}
-> status / partial_result / final_result
```

状态：

```text
queued
running
waiting_user
waiting_tool
need_confirmation
succeeded
failed
partial
cancelled
```

如流机器人可以先返回：

```text
任务已开始，workflow_id=xxx。我会在完成后推送结果。
```

Web 前端可以轮询或使用 SSE/WebSocket 查看进度。

## 13. 分阶段实施计划

### Phase 1: Auth Context + Tool Safety Foundation

目标：先建立企业 Agent 的安全地基，再做复杂 workflow。

内容：

- 基础 auth context：`tenant_id / user_id / groups / role`。
- AgentState 注入 auth context。
- tool metadata：read_only、side_effect、requires_confirmation、idempotency_required。
- confirmation flow 基础协议。
- idempotency key 生成与 tool execution record。
- timeout unknown 状态。
- side-effect eval。
- trace/debug 脱敏基础规则。

注意：细粒度文档 ACL 和跨部门 DLS 可以后续增强，但基础身份上下文必须从第一阶段
开始贯穿所有 Agent。

### Phase 2: Workflow Agent MVP

目标：在不破坏现有 RAG/chat/tool 的基础上新增 `workflow_agent`。

内容：

- Planner Agent。
- Workflow state。
- Mock enterprise tools。
- Verification Agent。
- Response Composer。
- Workflow eval cases。
- 使用 Phase 1 的 confirmation/idempotency 能力生成工单草稿或创建确认。

推荐 demo：

```text
帮我分析 payment-service 最近 30 分钟 5xx 增加的原因，并给一个排查工单草稿。
```

### Phase 3: Multi-hop RAG

内容：

- query decomposition。
- evidence group。
- gap detection。
- controlled iterative retrieval。
- multi-hop eval。

### Phase 4: MCP Adapter

内容：

- MCP tool registry。
- tool result adapter。
- large result context selection。
- schema validation。
- audit log。

### Phase 5: Enterprise Security

内容：

- document ACL。
- retrieval pre-filter。
- trace/debug 脱敏。
- unauthorized query audit。
- prompt injection via RAG policy。
- 高敏文档不暴露存在性策略。

### Phase 6: Federated Agent

内容：

- department adapter protocol。
- federated query planner。
- RRF fusion。
- partial result。
- cross-dept eval。

## 14. 当前项目映射

当前已有：

- FastAPI API 层。
- LangGraph 主图。
- chat/rag/tool/novel_script agents。
- SQLite checkpoint。
- session runtime。
- conversation history SQLite。
- Chroma vector memory。
- knowledge catalog。
- FTS5 lexical retrieval。
- Chroma dense retrieval。
- hybrid retrieval。
- rerank。
- citation。
- eval harness。
- LangSmith tracing。
- Docker deployment。

待新增：

- workflow_agent。
- planner/verifier/composer。
- workflow state。
- tool execution record。
- confirmation/idempotency。
- MCP adapter。
- multi-hop retrieval。
- enterprise ACL。
- federated gateway。

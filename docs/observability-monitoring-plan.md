# 项目监控与可观测性方案 v2

> v2 修订点：补齐 LLM/Embedding/Rerank provider 监控、Cost 指标、Eval 持续监控；明确 metrics backend 多 worker 方案；新增 PR-0 常量与契约层；硬化 label 基数与隐私规则；调整 Knowledge segment-size 监控为 P0。详见 §15。

## 1. 背景与目标

当前项目已经具备 FastAPI 服务、LangGraph 多节点工作流、RAG 检索链路、Tool Calling、文档导入、Eval Harness、LangSmith tracing、Docker 部署、SQLite checkpoint 和知识库 health 检查等能力。下一步监控建设的目标不是简单增加日志，而是形成一套面向企业级 Agent 的可观测闭环：

- 服务是否稳定：接口是否可用、延迟是否异常、错误率是否升高。
- Agent 是否可控：路由、规划、工具调用、确认流、幂等执行是否符合预期。
- RAG 是否可信：检索、过滤、重排、引用、fallback 哪一环出了问题。
- LLM/Embedding/Rerank Provider 是否健康：rate limit、provider 切换、retry、stream 中断是否被及时发现。
- 知识库是否健康：SQLite / FTS5 / Chroma 是否一致，索引是否膨胀或缺失。
- 成本是否可管理：token、LLM/embedding/rerank 调用次数与估算费用是否可量化、可归因。
- 问题是否可定位：从线上 bad case 能快速还原 route、node、retrieval、tool、answer 的完整链路。
- 迭代是否有据：eval pass rate、coverage、citation、fallback 等关键质量指标的趋势可持续追踪。

监控体系需要同时服务三类对象：

1. **线上运维**：快速发现服务故障、依赖异常、磁盘风险和延迟抖动。
2. **RAG/Agent 迭代**：通过指标定位召回、rerank、citation、planner、tool 等模块问题。
3. **面试与项目表达**：能清楚说明从 tracing、metrics、eval 到 feedback loop 的工程化闭环。

## 2. 总体架构

```text
FastAPI / Chat API / Knowledge API
        |
        +-- Request Metrics (multi-worker safe)
        +-- JSON Structured Logs (redacted)
        +-- LangSmith Trace (input/output redactor on)
        |
        v
LangGraph Runtime
        |
        +-- Node Timing
        +-- Route / Plan / Verify Metrics
        +-- Tool Metrics
        +-- Memory Metrics
        |
        v
RAG Pipeline
        |
        +-- Dense / Lexical / Hybrid / Rerank / Merge Metrics
        +-- Citation Metrics
        +-- Multi-hop Coverage Metrics
        |
        v
External Providers (LLM / Embedding / Rerank)
        |
        +-- Call / Latency / TTFT / Token / Cost
        +-- Rate Limit / Retry / Fallback
        |
        v
Knowledge Storage
        |
        +-- SQLite Catalog / FTS5 / Chroma
        +-- Segment Size / Consistency Health
        |
        v
Metrics Backend + Log Backend + Trace Backend
        |
        +-- Grafana / 公司内部 Dashboard
        +-- LangSmith
        +-- Eval Reports (nightly pushed as gauges)
        +-- Alerting
```

三层落地原则：

- **在线 metrics**：轻量、稳定、低成本、多 worker 安全，用于 dashboard 和告警。
- **结构化 debug payload**：当前接口已经返回 `debug.nodes`，继续作为单请求诊断的主入口。
- **离线 eval metrics**：保留 `scripts/eval_chat.py` 的 route / retrieval / citation / workflow / multi-hop 指标，用于策略回归和 bad case 分析；nightly 将关键聚合值推送回在线 gauge，形成质量趋势监控。

可观测性 sink 一律遵守 **fire-and-forget**：metrics、log、trace 任何写入异常必须吞掉并降级，不允许抛回主链路；同时上报 `observability_emit_failed_total{sink}` 自监控指标。

> 自监控悖论与兜底：当 metrics sink 本身挂掉时，`observability_emit_failed_total` 自身也可能发不出去。`emit` 包装器必须遵循三级降级：
>
> 1. **首选**：写 metrics counter。
> 2. **失败兜底**：立刻写一条结构化 error log（`event=observability.emit_failed`，含 `sink / error_type / dropped_metric_name`）到本地 stdout / 文件，保证审计可追溯。
> 3. **恢复观察**：metrics sink 恢复后通过 log → metrics 的延迟比对（log 计数 vs counter 增量）排查丢点期间。

## 3. 监控分层

### 3.1 服务层监控

目标：回答“服务有没有活着、是否变慢、是否报错”。

核心指标：

| 指标                                | 类型      | 维度                                 | 说明                                                                                                                                                                                  |
| ----------------------------------- | --------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `http_requests_total`               | counter   | `method,route_template,status_code`  | 接口请求量                                                                                                                                                                            |
| `http_request_duration_ms`          | histogram | `method,route_template,status_code`  | 接口耗时，重点看 p50/p95/p99                                                                                                                                                          |
| `http_5xx_rate`                     | derived   | `route_template`                     | 5xx 错误率                                                                                                                                                                            |
| `health_liveness_status`            | gauge     | `version,git_sha`                    | `/health` 是否正常                                                                                                                                                                    |
| `health_readiness_status`           | gauge     | `check`                              | `/health/ready` 依赖检查是否正常                                                                                                                                                      |
| `app_version_info`                  | gauge     | `version,git_sha,build_time`         | 当前运行版本                                                                                                                                                                          |
| `observability_emit_failed_total`   | counter   | `sink`                               | 自监控：metrics/log/trace 写入失败                                                                                                                                                    |
| `observability_label_dropped_total` | counter   | `metric_name,label_name,drop_reason` | 自监控：label 取值命中基数上限或白名单被丢弃；`metric_name` 必须来自 `app/constants/metrics.py` 注册表，`drop_reason` 取自枚举（`cardinality_overflow / not_whitelisted / redacted`） |

> `route_template` 必须是 FastAPI 注册路径模板（例如 `/knowledge/docs/{doc_id}/reindex`），**禁止**直接使用包含路径参数的原始 URL（如 `/knowledge/docs/abc123/reindex`），否则 `doc_id` 会把 path 维度打成高基数。FastAPI middleware 应通过 `request.scope["route"].path` 获取模板，未匹配到路由时 fallback 为 `__no_route__`。

现有基础：

- `/health` 已返回 `version / git_sha / build_time`。
- `/health/ready` 已检查 LLM profile、history path、Chroma dir。
- `scripts/smoke_docker.sh` 已覆盖 Docker restart 后 health 和 session 持久化。

建议扩展：

- 增加 FastAPI middleware，统一记录 request start / completed / failed。
- 把 `request_id / session_id / user_id_hash / tenant_id / route` 写入结构化日志。
- 对 `/chat`、`/knowledge/*`、`/health/ready` 分别看延迟和错误率。

### 3.2 LangGraph 工作流监控

目标：回答“Agent 走了哪条链路、哪个节点慢、是否发生降级或误路由”。

核心指标：

| 指标                         | 类型      | 维度                       | 说明                            |
| ---------------------------- | --------- | -------------------------- | ------------------------------- |
| `graph_request_total`        | counter   | `actual_route,intent_code` | Graph 请求量                    |
| `graph_node_duration_ms`     | histogram | `node,route`               | 各节点耗时                      |
| `graph_node_error_total`     | counter   | `node,error_code`          | 节点异常                        |
| `supervisor_route_total`     | counter   | `route,route_reason_code`  | Supervisor 路由分布             |
| `planner_schema_valid_total` | counter   | `valid,task_type`          | Planner 输出 schema 是否合法    |
| `workflow_status_total`      | counter   | `status,task_type`         | workflow 成功 / 部分成功 / 失败 |
| `workflow_degraded_total`    | counter   | `degrade_reason_code`      | 降级原因                        |
| `composer_passthrough_total` | counter   | `mode`                     | Composer 直通或合成策略         |

> `intent_code / error_code / route_reason_code / degrade_reason_code` 等所有 `*_code` / `*_reason` / `*_type` 类 label **必须来自 `app/constants/observability.py` 的封闭枚举**，禁止直接塞 LLM 自然语言解释（例如 Supervisor 的"用户在询问关于多个文档的对比，因此选择 multi_hop"）。原文必须落到日志/trace，metrics 上只允许出现枚举码（如 `multi_doc_compare`、`fallback_no_evidence`、`schema_invalid`）。新增取值前必须先在常量枚举中登记。

现有基础：

- `debug.nodes.*.sub_timings_ms` 已暴露节点内部耗时。
- `node_timings` 已能看到 supervisor / rag / merge / memory 等节点耗时。
- Phase 2 / Phase 3 已有 workflow 与 multi-hop eval 指标。
- LangSmith trace 已接入，可追踪单次链路。

建议扩展：

- 把 `node_timings` 同步为在线 histogram，而不只存在 debug payload。
- 对 `workflow_status=PARTIAL`、`degraded=True` 单独计数。
- 对 `multi_hop_agent` 记录 `hop_count / global_coverage / per_subquery_coverage / decompose_failed`。

### 3.3 RAG 检索质量监控

目标：回答“为什么答案错，是没召回、过滤误杀、rerank 错、引用错，还是模型没按证据回答”。

核心指标（histogram 数量类用 `_results` 后缀，避免与 Prometheus 自动生成的 `_count` 冲突）：

| 指标                            | 类型      | 维度                    | 说明                |
| ------------------------------- | --------- | ----------------------- | ------------------- |
| `rag_doc_used_total`            | counter   | `used,query_type`       | 是否使用知识库      |
| `rag_retrieval_empty_total`     | counter   | `collection,query_type` | 检索 0 命中         |
| `rag_dense_results`             | histogram | `collection`            | dense 返回数量      |
| `rag_lexical_results`           | histogram | `collection`            | lexical 返回数量    |
| `rag_hybrid_results`            | histogram | `collection`            | hybrid 后数量       |
| `rag_filtered_results`          | histogram | `query_type`            | threshold 后数量    |
| `rag_rerank_skipped_total`      | counter   | `reason`                | rerank 跳过原因     |
| `rag_citation_results`          | histogram | `query_type`            | 引用数量            |
| `rag_answer_has_citation_total` | counter   | `has_citation`          | 答案是否带引用      |
| `rag_fallback_total`            | counter   | `reason`                | 资料不足 / fallback |

Eval 指标继续保留：

- `top_k_hit`
- `filtered_hit`
- `rerank_hit`
- `merged_hit`
- `citation_hit`
- `citation_all_expected_docs_hit`
- `answer_has_citation`
- `citation_refs_valid`
- `fallback_accuracy`

在线与离线的边界：

- 在线不应每次都跑昂贵的 claim-level verification。
- 在线记录轻量指标：命中数量、引用数量、fallback、query_type、耗时。
- 离线 eval / 抽样任务再跑更重的 faithfulness、answer quality、claim-level 检查。
- nightly eval 完成后，将 `eval_pass_rate{suite,case_class}`、`eval_avg_hop_count`、`eval_avg_per_subquery_coverage` 等聚合值推送为在线 gauge，与在线 dashboard 同图比对。

### 3.4 Multi-hop 监控

目标：回答“复杂问题是否真的被拆解、每个子问题是否有证据、是否因为预算降级”。

核心指标：

| 指标                               | 类型      | 维度             | 说明                                                  |
| ---------------------------------- | --------- | ---------------- | ----------------------------------------------------- |
| `multi_hop_request_total`          | counter   | `status`         | multi-hop 请求量                                      |
| `multi_hop_decompose_duration_ms`  | histogram | -                | query decomposition 耗时                              |
| `multi_hop_hop_count`              | histogram | -                | 实际 hop 次数                                         |
| `multi_hop_subquery_count`         | histogram | -                | 子问题数量                                            |
| `multi_hop_global_coverage`        | histogram | -                | 全局覆盖度                                            |
| `multi_hop_per_subquery_coverage`  | histogram | -                | 子问题平均覆盖度                                      |
| `multi_hop_budget_exceeded_total`  | counter   | `budget_type`    | hop / chunk / time 预算超限                           |
| `multi_hop_degrade_total`          | counter   | `degrade_reason` | 降级原因                                              |
| `multi_hop_decompose_failed_total` | counter   | `error_code`     | decompose 失败原因（来自 PR-8.3 surfaced error code） |

现有基础：

- `debug_info.multi_hop` 已暴露 `hop_count / per_subquery_coverage / global_coverage / degrade_reason`。
- Eval summary 已有 `avg_hop_count / avg_global_coverage / avg_per_subquery_coverage`，PR-8.4 baseline：`avg_hop_count=2.25`、`avg_per_subquery_coverage=0.927`、`avg_global_coverage=0.925`。
- PR-8.3 已 surface `mh_decompose_error_code`，可直接作为 metric label。

建议扩展：

- 在线指标只记录聚合值，不保存全文 evidence。
- 保持 `evidence_groups_preview` preview-only，避免把全文塞进监控系统。
- 对 `decompose_failed`、`budget_exceeded`、`evidence_empty` 单独看趋势。

### 3.5 LLM Provider 监控（v2 新增）

目标：回答“LLM 是不是被限流、换没换 provider、token 花了多少钱、首字延迟是不是变长”。

核心指标：

| 指标                           | 类型      | 维度                                 | 说明                                            |
| ------------------------------ | --------- | ------------------------------------ | ----------------------------------------------- |
| `llm_call_total`               | counter   | `provider,model,status`              | LLM 调用数量                                    |
| `llm_call_duration_ms`         | histogram | `provider,model,phase`               | `phase=ttft\|total`，TTFT 监控流式首字延迟      |
| `llm_token_total`              | counter   | `provider,model,direction`           | `direction=in\|out`                             |
| `llm_cost_estimated_usd_total` | counter   | `provider,model,route,price_version` | 按 token × 单价估算成本（estimate，非账单口径） |
| `llm_rate_limited_total`       | counter   | `provider,model`                     | 429 / quota 命中                                |
| `llm_retry_total`              | counter   | `provider,reason`                    | 重试                                            |
| `llm_provider_fallback_total`  | counter   | `from_provider,to_provider`          | provider 降级切换                               |
| `llm_stream_aborted_total`     | counter   | `provider,reason`                    | 流式异常中断                                    |

现有基础：

- 项目曾因 GLM 429 被迫切换 provider，需要在线告警，避免人工发现。
- LangSmith 已记录单次调用，但聚合 cost / TTFT 必须有 metrics。

> 成本指标语义：`llm_cost_estimated_usd_total` **只是 estimate**，非 provider 账单口径。指标名带 `estimated`，并强制带 `price_version` label（如 `2026q2`）；price 表（`(provider, model) → unit_price_usd_per_1k_token`）维护在 `app/constants/llm_pricing.py`，每次调价升 `price_version`，老版本曲线保持可读。币种通过 metric name 后缀固定为 `_usd_total`；如需多币种另起 `_cny_total` 等独立指标，不在 label 上扩展，避免聚合错误。

### 3.6 Embedding / Rerank Provider 监控（v2 新增）

目标：回答“embedding/rerank 是不是限流、维度是不是变了、版本是不是飘了”。

核心指标：

| 指标                           | 类型      | 维度                                     | 说明            |
| ------------------------------ | --------- | ---------------------------------------- | --------------- |
| `embedding_call_total`         | counter   | `provider,model,status`                  | embedding 调用  |
| `embedding_call_duration_ms`   | histogram | `provider,model`                         | 调用耗时        |
| `embedding_token_total`        | counter   | `provider,model`                         | embedding token |
| `embedding_dim_mismatch_total` | counter   | `provider,model,expected_dim,actual_dim` | 维度异常        |
| `embedding_rate_limited_total` | counter   | `provider`                               | 限流            |
| `rerank_call_total`            | counter   | `provider,model,status`                  | rerank 调用     |
| `rerank_call_duration_ms`      | histogram | `provider,model`                         | 调用耗时        |
| `rerank_skipped_total`         | counter   | `reason`                                 | gate 跳过       |

`embedding_dim_mismatch_total` 不为零应直接 P0 告警，否则会静默打挂 Chroma 检索。

### 3.7 Tool Calling 与安全监控

目标：回答“工具有没有乱调、是否越权、确认流和幂等是否可靠”。

核心指标：

| 指标                                | 类型      | 维度                         | 说明                                            |
| ----------------------------------- | --------- | ---------------------------- | ----------------------------------------------- |
| `tool_call_total`                   | counter   | `tool_name,tool_type,status` | 工具调用数量                                    |
| `tool_call_duration_ms`             | histogram | `tool_name`                  | 工具调用耗时                                    |
| `tool_timeout_total`                | counter   | `tool_name`                  | 工具超时                                        |
| `tool_validation_failed_total`      | counter   | `tool_name,reason`           | 参数校验失败                                    |
| `tool_confirmation_required_total`  | counter   | `tool_name`                  | 需要确认的副作用工具                            |
| `tool_confirmation_replayed_total`  | counter   | `tool_name,status`           | 确认 token 重放结果                             |
| `tool_idempotency_hit_total`        | counter   | `tool_name`                  | 幂等命中                                        |
| `verifier_block_total`              | counter   | `reason,risk_level`          | Verifier 阻断                                   |
| `auth_denied_total`                 | counter   | `resource_type,reason`       | 权限拒绝                                        |
| `security_token_leak_suspect_total` | counter   | `source`                     | 自检：trace / checkpoint 中扫到 token-shaped 串 |

监控原则：

- `read_only` 工具允许自动执行，但需要记录调用和结果摘要。
- `side_effect` 工具必须经过 confirmation + idempotency。
- confirmation token 不应进入 checkpoint / trace 的长期状态，只记录脱敏摘要。token 泄漏自检分两期落地，避免一上来就跨外部系统：
  - **第一期（PR-6 内置）**：定期扫描本地 checkpoint DB / conversation history 文件，命中 token-shaped 串则 `security_token_leak_suspect_total{source="checkpoint|history"} +1` 并 P0 告警。范围可控，误报可调。
  - **第二期（离线审计任务）**：对 LangSmith trace 做抽样审计（按 trace*id 采样而非全量），结果落到独立 `security_audit*\*` 指标与日志，避免给 LangSmith 加重负载或跨系统鉴权风险。
- idempotency key 不应包含每次都变化的 `request_id`。

### 3.8 记忆与状态监控

目标：回答“会话状态是否可恢复、记忆是否污染、不同用户是否串数据”。

核心指标：

| 指标                               | 类型      | 维度                 | 说明               |
| ---------------------------------- | --------- | -------------------- | ------------------ |
| `checkpoint_write_total`           | counter   | `status`             | checkpoint 写入    |
| `checkpoint_restore_total`         | counter   | `status`             | checkpoint 恢复    |
| `conversation_history_write_total` | counter   | `status`             | 历史记录写入       |
| `memory_vector_write_total`        | counter   | `status,skip_reason` | 长期记忆写入       |
| `memory_retrieval_results`         | histogram | `policy`             | 记忆召回数量       |
| `memory_write_skip_total`          | counter   | `skip_reason`        | 记忆跳过原因       |
| `session_state_size_bytes`         | histogram | -                    | session state 大小 |

需要重点监控：

- `session_id / tenant_id / user_id_hash` 是否始终存在。
- memory 查询是否带正确 `where` 过滤。
- RAG 文档命中类回答是否被跳过写入 vector memory，避免把知识库事实重复写成用户记忆。

### 3.9 知识库健康监控

目标：回答“知识库索引是否一致、是否存在幽灵 chunk、Chroma 是否异常膨胀”。

核心指标：

| 指标                                   | 类型    | 维度                          | 说明                    |
| -------------------------------------- | ------- | ----------------------------- | ----------------------- |
| `knowledge_document_count`             | gauge   | `backend=sqlite`              | 文档数量                |
| `knowledge_chunk_count`                | gauge   | `backend=sqlite\|fts\|chroma` | chunk 数量              |
| `knowledge_missing_chroma_chunk_count` | gauge   | -                             | SQLite 有但 Chroma 缺失 |
| `knowledge_orphan_chroma_chunk_count`  | gauge   | -                             | Chroma 有但 SQLite 缺失 |
| `knowledge_import_total`               | counter | `status,source_type`          | 导入数量                |
| `knowledge_import_skipped_total`       | counter | `reason`                      | 幂等跳过                |
| `knowledge_reindex_total`              | counter | `scope,status`                | reindex 执行            |
| `chroma_segment_total_size_bytes`      | gauge   | -                             | Chroma 目录总大小       |
| `chroma_largest_segment_size_bytes`    | gauge   | -                             | 最大 segment 文件       |
| `chroma_avg_bytes_per_chunk`           | gauge   | -                             | 平均每 chunk 占用       |
| `chroma_segment_size_anomaly_total`    | counter | `reason`                      | segment 异常            |

现有基础：

- `/knowledge/health` 已检查 SQLite / FTS / Chroma chunk count 一致性。
- Chroma eval bloat 事故已明确需要补 segment size 检查（曾出现过 segment 膨胀至 ~250GB），故 v2 中升级为 **P0** 并前置到 PR-2。
- 导入已支持 `content_hash` 幂等，避免不变文档重复 delete / upsert。

建议扩展：

- `/knowledge/health` 增加 `storage` 字段：

```json
{
  "storage": {
    "chroma_persist_dir": "data/chroma",
    "total_size_bytes": 123456,
    "largest_file_bytes": 45678,
    "avg_bytes_per_chunk": 1024,
    "segment_size_anomaly": false
  }
}
```

- warning 类型统一常量化，例如：
  - `chroma_count_mismatch`
  - `missing_chroma_chunks`
  - `orphan_chroma_chunks`
  - `chroma_segment_size_anomaly`
  - `fts_count_mismatch`

### 3.10 Eval 持续监控（v2 新增）

目标：回答“最近一次发布有没有让 RAG/Agent 质量退化”。

核心指标（由 nightly CI / 手工 eval 推送）：

| 指标                             | 类型      | 维度                   | 说明                                                                           |
| -------------------------------- | --------- | ---------------------- | ------------------------------------------------------------------------------ |
| `eval_pass_rate`                 | gauge     | `suite,case_class`     | pass rate（每次 push 覆盖；run_id 不进 label，靠时间序列天然区分每次 run）     |
| `eval_avg_hop_count`             | gauge     | `suite`                | multi-hop 平均 hop 数                                                          |
| `eval_avg_per_subquery_coverage` | gauge     | `suite`                | 平均子问题覆盖度                                                               |
| `eval_avg_global_coverage`       | gauge     | `suite`                | 平均全局覆盖度                                                                 |
| `eval_decompose_failed_total`    | counter   | `error_code`           | decompose 失败计数                                                             |
| `eval_run_duration_ms`           | histogram | `suite`                | eval 单次运行耗时                                                              |
| `eval_run_status`                | gauge     | `suite`                | 上一次 eval 是否成功（1/0）                                                    |
| `eval_last_success_timestamp`    | gauge     | `suite`                | 上一次 eval 成功的 unix timestamp（秒）                                        |
| `eval_staleness_seconds`         | gauge     | `suite`                | 当前时间 − 上次成功时间，由查询端表达式计算                                    |
| `eval_run_info`                  | gauge     | `suite,git_sha,run_id` | 元数据信息指标，值恒为 1，仅承载 run_id / git_sha，便于在 dashboard / 日志关联 |

> Stale 处理与基数控制：push gateway / OTLP gauge 不会自动过期，长期残留的"上一次成功 pass_rate"会让 dashboard 误以为质量仍健康。规则：
>
> 1. 质量类 gauge（`eval_pass_rate / eval_avg_*`）只保留 `suite,case_class` 这种低基数维度，**不允许**把 `eval_run_id` 作为它们的 label——否则 Prometheus / TSDB 中会按 run_id 不断 churn 出新 series，长期形成基数爆炸。每次 run 推送时直接覆盖（覆盖语义由时间序列本身表达）。
> 2. `eval_run_id`（如 `2026-05-15T03:00Z`）只放在两处：(a) 推送侧的日志/审计记录，(b) 单独的 `eval_run_info{suite,git_sha,run_id}=1` 元数据指标，用 PromQL `*` 关联即可在 dashboard 上把当前曲线还原回具体 run。元数据指标推送时使用 `delete_on_failure` 或仅保留最近 N 次。
> 3. 每次 eval 推送都更新 `eval_last_success_timestamp{suite}`。Dashboard 与 P1 告警必须叠加 staleness 守门：当 `time() − eval_last_success_timestamp{suite} > 36h`（按 nightly 周期 + 缓冲）触发 `eval_pipeline_stale` 告警，质量类指标在 dashboard 上同步置灰。

## 4. 指标命名与契约规范

集中常量与契约（PR-0 必须先落地）：

```text
app/constants/metrics.py        # metric name / label name / event name 枚举
app/constants/observability.py  # bucket 边界、label 基数上限、redactor 接口
```

命名约定：

- counter 使用 `_total` 结尾。
- histogram 数量类用 `_results`、`_size`，时间类用 `_duration_ms`，字节类用 `_bytes`，明确单位。
- gauge 表达当前状态，例如 chunk count、segment size、最近一次 eval pass rate。
- label 名称固定使用 snake_case。
- 不把高基数字段直接作为 label，例如 `request_id`、完整 `query`、完整 `doc_id` 不进入 metrics label。

Histogram bucket 必须显式声明（不使用默认 bucket），建议：

- 时间类（ms）：`[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000]`
- 数量类（results）：`[0, 1, 2, 5, 10, 20, 50, 100]`
- 字节类：`[1KB, 10KB, 100KB, 1MB, 10MB, 100MB, 1GB]`

推荐 label：

- `route`
- `node`
- `query_type`：`definition / comparison / followup / fallback / factual / multi_hop`
- `tool_name`
- `tool_type`：`read_only / side_effect`
- `status`
- `reason` / `error_code`
- `tenant_id`（受基数保护，见下）
- `provider` / `model`
- `env`
- `version`

Label 基数上限（硬性规则）：

- 单 metric 单 label 取值数 > **1000** 一律不允许直接进 metrics，必须降级为日志/trace。
- `user_id`：禁止进 metrics；日志和 trace 使用 `user_id_hash`（截短哈希）。
- `doc_id` / `chunk_id`：只用于 debug / eval / trace，不进在线 metrics label。
- `session_id` / `request_id`：只用于日志和 trace。
- `tenant_id`：当不同租户数 ≤ 100 时直接作为 label；超出时聚合为 `tenant_bucket`（如按租户级别 / 行业 hash 分桶）。
- `tool_name`：限定为白名单内的注册工具名，外部动态生成的工具必须先注册。

### 4.1 在线 ↔ 离线 ↔ debug payload 字段映射

下表用于排查在线 dashboard 与 eval / debug 数字对不上时的快速对齐（避免 v1 三方各写各的）：

| 概念             | 在线 metric                                    | eval 字段                 | debug payload 字段                      |
| ---------------- | ---------------------------------------------- | ------------------------- | --------------------------------------- |
| 是否使用知识库   | `rag_doc_used_total{used}`                     | `doc_used`                | `debug.rag.doc_used`                    |
| 检索原始命中     | `rag_dense_results`                            | `dense_count`             | `debug.rag.dense_count`                 |
| 词法命中         | `rag_lexical_results`                          | `lexical_count`           | `debug.rag.lexical_count`               |
| 阈值过滤后       | `rag_filtered_results`                         | `filtered_count`          | `debug.rag.filtered_count`              |
| 引用数量         | `rag_citation_results`                         | `citation_count`          | `debug.rag.citation_count`              |
| 是否带引用       | `rag_answer_has_citation_total`                | `answer_has_citation`     | `debug.rag.answer_has_citation`         |
| Multi-hop hop 数 | `multi_hop_hop_count`                          | `hop_count`               | `debug.multi_hop.hop_count`             |
| 子问题覆盖       | `multi_hop_per_subquery_coverage`              | `per_subquery_coverage`   | `debug.multi_hop.per_subquery_coverage` |
| 全局覆盖         | `multi_hop_global_coverage`                    | `global_coverage`         | `debug.multi_hop.global_coverage`       |
| Decompose 失败   | `multi_hop_decompose_failed_total{error_code}` | `mh_decompose_error_code` | `debug.multi_hop.decompose_error_code`  |

字段名一旦在 PR-0 落地，三方必须同步演进；新增字段先在常量层登记，再在三处实现。

### 4.2 Metric Label 白名单（v2 新增）

PR-0 在 `app/constants/observability.py` 落地以下白名单作为单一事实来源，PR-1+ 的所有 metrics 必须遵守，CI lint 校验未登记的 label / 取值不允许出现：

| 类别                                                                     | 规则                                                                                                                                                           |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `route_template`                                                         | 必须是 FastAPI 注册的路径模板（如 `/knowledge/docs/{doc_id}/reindex`），未匹配 fallback `__no_route__`；禁止使用原始 URL                                       |
| `*_code` / `*_reason` / `*_type`                                         | 必须来自封闭枚举（`route_reason_code / error_code / degrade_reason_code / intent_code / skip_reason / decompose_error_code` 等），原始自然语言只能进 log/trace |
| `tenant_id`                                                              | 仅当全局租户数 ≤ 100 直接作为 label；超出走 `tenant_bucket`（hash 后取桶号），原始 id 仅进日志                                                                 |
| `user_id`                                                                | 永远禁止进 metrics；trace/log 使用 `user_id_hash` 截短哈希                                                                                                     |
| `request_id` / `session_id` / `doc_id` / `chunk_id` / `query` / `prompt` | 全部禁止进 metrics label，仅允许进日志和 trace                                                                                                                 |
| `tool_name`                                                              | 限定为注册白名单内的工具名；动态生成的工具未注册不允许调用                                                                                                     |
| `provider` / `model`                                                     | 来自 provider 注册表，未注册的 provider 必须先登记                                                                                                             |
| 通用基数硬上限                                                           | 单 metric × 单 label 取值数 > **1000** 必须降级到 log/trace；CI lint 在埋点装饰器层做静态校验                                                                  |

> 实施要点：
>
> - 所有埋点统一通过 `app/observability/emit.py` 的 `emit_counter / emit_histogram / emit_gauge`，参数里的 label dict 在运行时再做一次基数兜底（命中上限自动 drop 该 label 取值并 +1 `observability_label_dropped_total`）。
> - 新增枚举值 = 改 `app/constants/observability.py` + 单测，禁止业务代码里裸字符串。

## 5. 日志规范

继续使用 JSON structured log，建议统一事件类型：

| event                         | 说明                                   |
| ----------------------------- | -------------------------------------- |
| `request.started`             | 请求开始                               |
| `request.completed`           | 请求完成                               |
| `request.failed`              | 请求异常                               |
| `graph.node.started`          | 节点开始                               |
| `graph.node.completed`        | 节点完成                               |
| `rag.retrieval.completed`     | 检索完成                               |
| `llm.call.completed`          | LLM 调用完成（含 provider/token/cost） |
| `embedding.call.completed`    | embedding 调用完成                     |
| `tool.call.started`           | 工具开始                               |
| `tool.call.completed`         | 工具完成                               |
| `knowledge.import.completed`  | 文档导入完成                           |
| `knowledge.health.checked`    | health 检查完成                        |
| `security.denied`             | 权限或 verifier 拦截                   |
| `security.token_leak_suspect` | 自检发现疑似 token 泄漏                |
| `observability.emit_failed`   | metrics/log/trace 写入失败             |

每条日志建议包含：

```json
{
  "ts": "2026-05-15T10:00:00Z",
  "level": "INFO",
  "event": "request.completed",
  "request_id": "xxx",
  "session_id": "xxx",
  "tenant_id": "default",
  "user_id_hash": "h:ab12...",
  "route": "rag_agent",
  "duration_ms": 1234,
  "status": "ok"
}
```

安全要求（硬性）：

- 不记录完整 prompt。
- 不记录完整文档 chunk。
- 不记录 confirmation token 明文。
- 不记录 API key、cookie、内部鉴权 header。
- evidence 只记录 preview 和 doc metadata。
- 任何疑似敏感字段统一走 `app/observability/redactor.py` 处理后再写入 sink。

## 6. Trace 规范

LangSmith 适合做单请求链路定位，不适合作为所有聚合监控的唯一来源。

Trace metadata 建议包含：

- `request_id`
- `session_id`
- `tenant_id`
- `user_id_hash`
- `route`
- `intent`
- `query_type`
- `workflow_status`
- `degrade_reason`
- `app_version`
- `git_sha`

节点 span 建议包含：

- `node_name`
- `duration_ms`
- `input_size`
- `output_size`
- `error_type`
- `retrieval_counts`
- `tool_name`
- `risk_level`

注意（硬性）：

- LangSmith 接入必须配置输入/输出 redactor（`LANGCHAIN_HIDE_INPUTS / LANGCHAIN_HIDE_OUTPUTS` 或自定义 redactor），禁止全量记录 prompt 和 retrieved chunks；redactor 与 §5 共用同一个实现，避免双份逻辑。
- side-effect 工具结果只放摘要。
- 大文档、大 JSON、大日志输出需要先做 context selection 或 preview 化。
- confirmation token、API key、内部 cookie 一律不允许进 trace。

## 7. 告警规则

### 7.1 P0 告警

| 规则                    | 条件                                                                                                                      | 处理建议                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 服务不可用              | `/health` 连续失败 3 次                                                                                                   | 检查进程、容器、端口、依赖                                                                        |
| Readiness 失败          | `/health/ready` 失败持续 3 分钟                                                                                           | 检查 API key、history path、Chroma dir                                                            |
| 磁盘风险                | Chroma segment 单文件 > 阈值 **且**（`avg_bytes_per_chunk` 较 7d baseline > 3× 或目录 7d 日增长 > 50%）；或磁盘使用 > 90% | 停止 eval / import，检查 Chroma bloat（口径以 PR-2 的复合规则为准；只命中单条降为 P1，详见 §7.2） |
| Embedding 维度异常      | `embedding_dim_mismatch_total > 0`                                                                                        | 立即停写入 Chroma，回滚 embedding model                                                           |
| LLM 全 provider 失联    | `llm_provider_fallback_total` 触发但仍失败                                                                                | 切换备用 provider 或停服降级                                                                      |
| 权限绕过风险            | `auth_denied_total` 异常下降且敏感工具调用上升                                                                            | 检查 ACL / verifier                                                                               |
| 副作用重复执行          | `tool_idempotency_hit_total` 异常或重复工单                                                                               | 检查 idempotency key                                                                              |
| Token 泄漏自检          | `security_token_leak_suspect_total > 0`                                                                                   | 立即下线相关 trace / checkpoint，回滚 redactor                                                    |
| Observability sink 异常 | `observability_emit_failed_total` 突增                                                                                    | 检查 metrics/log/trace backend                                                                    |

### 7.2 P1 告警

| 规则                | 条件                                                                                          | 处理建议                                                             |
| ------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 接口延迟升高        | `/chat` p99 > 15s 持续 5 分钟                                                                 | 看 node timing，定位 LLM / rerank / tool                             |
| LLM TTFT 升高       | `llm_call_duration_ms{phase=ttft}` p95 倍增                                                   | 检查 provider 状态、网络、模型切换                                   |
| 5xx 升高            | 5xx rate > 2% 持续 5 分钟                                                                     | 查 request.failed 和 exception                                       |
| 检索 0 命中升高     | `rag_retrieval_empty_total` 比例突增                                                          | 检查 Chroma 路径、embedding provider、索引一致性                     |
| LLM 限流升高        | `llm_rate_limited_total` 持续上升                                                             | 切换 provider 或调速                                                 |
| citation 下降       | `answer_has_citation` 或 `citation_hit` 下降                                                  | 检查 context / prompt / citation builder                             |
| workflow 降级升高   | `workflow_degraded_total` 突增                                                                | 检查 planner、tool、verifier、multi-hop budget                       |
| Eval 趋势退化       | `eval_pass_rate` 较上一次下降 > 3pp                                                           | 暂停发布，定位 bad case                                              |
| Chroma 单一信号异常 | 仅命中"单文件 > 阈值" 或仅命中"`avg_bytes_per_chunk` 偏离" 或仅命中"目录日增长率超阈"中的一项 | 进入观察窗口，触达第二条信号即升 P0；阈值与默认值见 PR-2             |
| Label 维度异常丢弃  | `rate(observability_label_dropped_total[10m]) > 0` 持续异常                                   | 排查埋点是否传入未注册 / 高基数 label，按 `drop_reason` 区分修复路径 |

### 7.3 P2 告警

| 规则                | 条件                                               | 处理建议                                      |
| ------------------- | -------------------------------------------------- | --------------------------------------------- |
| token 成本升高      | `llm_cost_estimated_usd_total` 单请求 p95 超过阈值 | 检查 multi-hop / context compression / prompt |
| rerank 跳过异常     | `rag_rerank_skipped_total` 分布变化                | 检查 rerank gate                              |
| memory 写入跳过异常 | `memory_write_skip_total` 分布突变                 | 检查 memory write policy                      |
| import skip 下降    | 相同文档导入但 `content_unchanged` skip 下降       | 检查 content_hash 幂等                        |

## 8. Dashboard 设计

建议先做 8 个看板（v2 增加 LLM Provider 与 Eval 趋势）。

### 8.1 服务总览

- QPS
- 2xx / 4xx / 5xx
- p50 / p95 / p99
- `/health` / `/health/ready`
- version / git_sha
- Docker restart / smoke 结果
- `observability_emit_failed_total` 自监控

### 8.2 Graph 工作流

- route 分布
- 各节点耗时
- workflow status
- degraded reason
- planner schema valid rate
- verifier block reason

### 8.3 RAG 质量

- doc_used rate
- retrieval empty rate
- dense / lexical / hybrid results
- filtered / rerank / merged results
- citation count
- fallback rate
- query_type 分布

### 8.4 Multi-hop

- multi-hop 请求量
- hop_count 分布
- subquery_count 分布
- global_coverage / per_subquery_coverage
- budget exceeded
- decompose failed by error_code

### 8.5 LLM / Embedding / Rerank Provider（v2 新增）

- 各 provider 调用量与 status 分布
- TTFT / total latency p50/p95/p99
- token in/out 趋势
- 估算成本（按 route / model）
- rate limit、retry、provider fallback
- embedding 维度异常 / rerank skip

### 8.6 Tool 与安全

- tool call count
- tool success / timeout / failed
- confirmation required / replayed
- idempotency hit
- auth denied
- verifier block
- token leak suspect

### 8.7 Knowledge Health

- document count
- chunk count: SQLite / FTS / Chroma
- missing / orphan chunk count
- Chroma segment total size / largest segment / avg bytes per chunk
- import / reindex success
- content_hash skip rate

### 8.8 Eval 趋势（v2 新增）

- pass_rate 时间序列（按 suite / case_class）
- avg_hop_count / avg_per_subquery_coverage / avg_global_coverage
- decompose_failed by error_code
- 上一次 eval 运行状态与耗时

## 9. 实现方案

### PR-0：可观测性常量与契约（v2 新增，必须先做）

目标：避免后续 PR 各自命名导致指标飘散。

改动建议：

- 新增 `app/constants/metrics.py`：metric name、label name、event name 全部枚举化。
- 新增 `app/constants/observability.py`：histogram bucket、label 基数上限、`query_type / tool_type / route` 枚举。
- 新增 `app/observability/redactor.py`：统一 prompt / chunk / token / header 脱敏接口，供 log、trace、metrics label 复用。
- 新增 `app/observability/emit.py`：fire-and-forget 包装层，捕获异常 → `observability_emit_failed_total +1`；label 维度命中基数上限或不在白名单时自动 drop 该取值并 +1 `observability_label_dropped_total{metric_name,label_name,drop_reason}`。
- 在线 ↔ 离线 ↔ debug payload 三方字段映射写入文档与单测，PR-1+ 必须遵守。

验收：

- 任何新指标 PR 必须先在 `metrics.py` 注册，未注册的字符串不允许在业务代码出现（lint 规则）。
- redactor 单测覆盖 prompt、chunk、API key、confirmation token 四类样本。
- 单测：构造 1001 个不同 `tenant_id` 触发 `cardinality_overflow`、构造未注册 `tool_name` 触发 `not_whitelisted`，`observability_label_dropped_total` 对应 reason 各 +1。

### PR-1：Metrics 基础设施

目标：建立统一入口，**直接选定多 worker 安全的 backend**，不再走"内存聚合 MVP"过渡态。

改动建议：

- 新增 `app/observability/metrics.py`
  - `Counter / Histogram / Gauge` 抽象。
  - 默认 backend = `prometheus_client` 多进程模式（设置 `PROMETHEUS_MULTIPROC_DIR`），uvicorn 多 worker 时聚合可见；备选 OTLP exporter 直接推到公司监控 SDK。
  - 不允许使用进程内独占 dict 作为生产 backend。
- Prometheus multiprocess 生命周期（必须明确，否则多 worker 会重复或残留）：
  - **启动前清空**：进程启动脚本（uvicorn 启动前 / Dockerfile entrypoint）`rm -rf $PROMETHEUS_MULTIPROC_DIR/* && mkdir -p $PROMETHEUS_MULTIPROC_DIR`，避免上次崩溃残留的 `*.db` 让计数翻倍。这一步是兜底主路径——SIGKILL / OOM / 容器被杀都不会跑任何退出钩子，必须假设上次运行不曾清理过自己。
  - **优雅退出清理**：通过 FastAPI / uvicorn 的 `worker_exit` 钩子或 `SIGTERM` handler 调用 `prometheus_client.multiprocess.mark_process_dead(pid)`，回收该 worker 的样本文件。仅覆盖优雅退出场景，不可作为主防线。
  - **stale pid 巡检**：周期任务（如每 30s）扫描 `$PROMETHEUS_MULTIPROC_DIR`，遇到 pid 已不存在的 `*.db` 调用 `mark_process_dead(pid)` 清理，覆盖 SIGKILL / 崩溃残留。
  - **dev / reload 模式**：`uvicorn --reload` 不允许启用 multiprocess 模式（reloader fork 出的子进程 pid 复用会污染样本），改用单进程 backend 或显式禁用 metrics，启动时 fail-fast 校验。
  - **健康检查**：定期巡检 `$PROMETHEUS_MULTIPROC_DIR` 文件数与 worker 数比值，异常时上报 `observability_multiproc_dir_anomaly_total`。
- FastAPI middleware：
  - 记录 request count、duration、status。
  - 将 `request_id` 放入 log context、trace metadata。
  - `route_template` 通过 `request.scope["route"].path` 取，未匹配置 `__no_route__`。
- 暴露 `/metrics`：Prometheus text exposition；多进程模式下使用 `multiprocess.MultiProcessCollector`。

验收：

- 启动 `uvicorn --workers 2`，向 `/chat` 发请求，`/metrics` 中能看到所有 worker 累计计数。
- 向某个 worker 发 `SIGTERM` 优雅停掉后，其样本文件被 `mark_process_dead` 清理，`/metrics` 不再含其残留。
- 向某个 worker 发 `SIGKILL`（模拟 OOM / `kill -9`）后，stale pid 巡检任务在下一周期内清理残留，并触发一次 `observability_multiproc_dir_anomaly_total +1`；不再出现累计翻倍。
- 进程整体重启场景：上次崩溃后启动脚本清空 multiproc dir，新一轮 `/metrics` 不含旧样本。
- `--reload` 模式下启用 multiprocess backend 必须启动失败。
- `/health/ready` 失败能记录 readiness status。
- 不记录 query 全文和敏感 header；`route_template` 不出现路径参数原值。

### PR-2：Knowledge Health Storage Metrics（v2 升级为 P0，前置）

目标：补齐 Chroma segment size 和索引一致性监控，杜绝 250GB 事故重演。本期前置是因为这条线对历史事故的回归收益最大、改动局限于 health 层、风险低。

改动建议：

- 扩展 `app/knowledge/health.py`：统计 Chroma persist dir 总大小、最大单文件、avg bytes per chunk；超阈值加入 warning。
- 扩展 `/knowledge/health` response schema（见 §3.9）。
- 将 health 结果同步为 gauge，并接入 P0 告警。
- 阈值默认值 + 可配置：
  - 默认：单 segment > **1GB** 或 `avg_bytes_per_chunk` 较 7d baseline 上涨 > **3×** 或目录总大小 7d 增长率 > **50%/day**。
  - 通过 `app/constants/knowledge_health.py` + 环境变量覆盖（`CHROMA_SEGMENT_SIZE_WARN_BYTES`、`CHROMA_AVG_BYTES_PER_CHUNK_RATIO`、`CHROMA_DIR_DAILY_GROWTH_RATIO`），大知识库可上调，避免误报。
  - 告警必须**同时命中**绝对阈值与增长率 / 平均值偏离才升 P0；只命中其中之一记 P1，避免单一阈值在大库场景误报。

验收：

- 构造异常 segment size 时返回 `chroma_segment_size_anomaly` 并告警。
- 大库默认阈值上调后告警不再触发，配置生效。
- SQLite / FTS / Chroma count mismatch 仍正常报告。

### PR-3：Graph Node Metrics

目标：把现有 `node_timings` 在线指标化。

改动建议：

- 在 graph runner 或 node wrapper 层统一记录：
  - `graph_node_duration_ms`
  - `graph_node_error_total{node,error_code}`
  - `graph_request_total{actual_route,intent_code}`
- 对 supervisor 输出记录：`supervisor_route_total{route,route_reason_code}`、`intent_code`，原始自然语言进 log/trace。
- 对 workflow 输出记录：`workflow_status_total`、`workflow_degraded_total{degrade_reason_code}`。

验收：

- 简单 chat / rag / workflow / multi-hop 请求能看到不同 route 计数。
- 节点异常时不会影响原异常抛出和 debug payload。
- 所有 `*_code` label 取值都来自 `app/constants/observability.py` 枚举，无自然语言渗入。

### PR-4：RAG + Multi-hop Metrics

目标：把 RAG pipeline 与 multi-hop 关键阶段暴露为在线指标。

改动建议：

- 在 `rag_agent` / retrieval pipeline finalize 处记录：dense_results、lexical_results、hybrid_results、filtered_results、citation_results、doc_used、fallback、rerank skipped reason。
- query classification 作为 label。
- multi-hop 单独记录 coverage 和 budget 与 `multi_hop_decompose_failed_total{error_code}`。

验收：

- 跑 baseline eval 后，在线指标和 eval summary 趋势一致（参考 §4.1 字段映射）。
- multi-hop 的 `filtered/rerank/merged` 不误计入单跳分母。

### PR-5：LLM / Embedding / Rerank Provider Metrics（v2 新增）

目标：让 provider 故障第一时间被监控发现，而不是事后人工换 provider。

改动建议：

- 在 LLM / embedding / rerank 客户端封装层统一埋点（通过 PR-0 的 emit 包装器）：
  - 调用计数、duration、TTFT（流式）、token in/out。
  - 限流、重试、provider fallback。
  - embedding 维度校验：实际维度 ≠ 期望维度时 `embedding_dim_mismatch_total +1` 并阻断写入。
- 成本估算：`app/constants/llm_pricing.py` 维护 `(provider, model) → unit_price_usd_per_1k_token` + `price_version`，调用结束累加 `llm_cost_estimated_usd_total{provider,model,route,price_version}`。

验收：

- 模拟 GLM 429，能看到 `llm_rate_limited_total` 与 `llm_provider_fallback_total` 同步上升。
- embedding 维度异常时 Chroma 不会被写入污染。
- dashboard 可按 route 拆 cost；调价升 `price_version` 后老曲线仍可读。

### PR-6：Tool / Security Metrics

目标：让工具调用、安全拦截、确认流可观测。

改动建议：

- Tool Agent：tool call count、duration、status、timeout；read_only / side_effect 分开统计。
- Verifier：block / allow / need_confirmation。
- Confirmation：token issued / replayed / expired / invalid。
- Idempotency：key hit / miss / replay result。
- Token 泄漏自检（第一期）：定时任务扫描本地 checkpoint DB / conversation history 文件，命中 token-shaped 串 → `security_token_leak_suspect_total{source="checkpoint|history"} +1` 并 P0；LangSmith 抽样审计放第二期独立离线任务。

验收：

- `ticket.create_draft` 或 mock side-effect 工具能看到 confirmation metrics。
- 重放相同 confirmation 不会重复 side-effect，且 idempotency hit 可见。
- 故意把 token 写入 checkpoint，下一周期内 `security_token_leak_suspect_total{source="checkpoint"} +1` 并触发 P0。

### PR-7：Eval 持续监控（v2 新增）

目标：把离线 eval 接进在线趋势，并避免 stale gauge 误导。

改动建议：

- `scripts/eval_chat.py` 增加 `--push-metrics` 选项：运行结束推送 §3.10 中所有 gauge / counter；质量类 gauge **不带** `eval_run_id` label（避免历史 series 基数爆炸），改为同步推送 `eval_run_info{suite,git_sha,run_id}=1` 元数据指标 + 更新 `eval_last_success_timestamp{suite}`，并在审计日志记录本次 `run_id`。
- nightly CI 任务调用上述命令，失败 → `eval_run_status=0` 并触发 P1 告警。
- Dashboard §8.8 直接消费这些 gauge；面板叠加 `time() − eval_last_success_timestamp > 36h` staleness 守门，超期置灰并触发 `eval_pipeline_stale` 告警。
- push gateway 配置 `delete_on_failure` 或周期清理脚本，仅保留最近 N 次 `eval_run_info`，避免元数据指标长期累积。

验收：

- nightly 运行后 dashboard `eval_pass_rate` 出现新数据点，且不会因 run_id 维度产生新 series；`eval_run_info` 中 `run_id` 更新。
- 故意停止 nightly 36h+ 后，dashboard 上 pass rate 面板置灰、staleness 告警触发。
- 故意制造退化 eval 时能看到 `eval_pass_rate` 下降并告警。

### PR-8：Dashboard 与 Alert Rules

目标：把指标变成可读的运营面板。

改动建议：

- 如果使用 Grafana：dashboard JSON + alert rule 配置，按 §8 的 8 个看板。
- 如果使用公司内部平台：输出指标清单、label 字典、阈值配置说明。
- 文档补充 runbook（见 §10）。

验收：

- 能通过 dashboard 定位一次故障属于服务、RAG、Tool、Knowledge、LLM Provider 还是 Eval 退化。

## 10. Runbook

### 10.1 检索突然 0 命中

排查顺序：

1. 看 `/knowledge/health`：SQLite / FTS / Chroma chunk count 是否一致。
2. 看 Chroma persist dir：是否指向临时空目录（注意 `.env` 加载时机问题）。
3. 看 `embedding_dim_mismatch_total` 是否非零。
4. 看 embedding provider / model 是否变更，`embedding_rate_limited_total` 是否突增。
5. 看 ACL filter 是否过严。
6. 看 query classification 是否误判到不检索路径。
7. 跑单 case eval，检查 `dense_count / lexical_count / topDocs`。

### 10.2 citation 下降

排查顺序：

1. 看 retrieval 是否命中正确 chunk。
2. 看 threshold 是否过滤掉正确 chunk。
3. 看 rerank 是否误杀。
4. 看 chunk merge 是否合并错。
5. 看 answer prompt 是否要求引用。
6. 看 citation builder 是否生成 ref。

### 10.3 p99 延迟升高

排查顺序：

1. 看 `graph_node_duration_ms` 找最慢节点。
2. 如果是 LLM，看 `llm_call_duration_ms{phase=ttft}` 与 `llm_rate_limited_total`，确认是否 provider 抖动或限流。
3. 如果是 RAG，拆 dense / lexical / rerank / answer generation。
4. 如果是 Tool，检查外部 API timeout。
5. 如果是 multi-hop，看 hop_count、decompose、budget。
6. 如果是 memory，看 vector write 或 history write。

### 10.4 Chroma 体积异常

排查顺序：

1. 看 `chroma_largest_segment_size_bytes` 和 `chroma_avg_bytes_per_chunk`。
2. 看最近是否反复导入相同文档。
3. 看 `knowledge_import_skipped_total{reason="content_unchanged"}` 是否正常。
4. 停止重复 eval / import。
5. 必要时重新构建隔离 Chroma persist dir，再切换。

### 10.5 side-effect tool 超时未知

排查顺序：

1. 看 tool timeout 日志。
2. 用 idempotency key 查询是否已有执行结果。
3. 如果工具端支持 query-by-idempotency-key，先查后重试。
4. 如果状态未知，返回用户"操作状态未知，请确认后重试"，不要盲目重复创建。

### 10.6 LLM provider 异常（v2 新增）

排查顺序：

1. 看 `llm_call_total{status}` 与 `llm_rate_limited_total` 哪个 provider 出问题。
2. 看是否已自动 `llm_provider_fallback_total` 切换；若未切换，手动切换备用 provider。
3. 看 cost：是否突增（可能误用大模型）。
4. 看 stream 中断比例 `llm_stream_aborted_total`。
5. 必要时降级到非流式或缩短 prompt。

### 10.7 Eval 趋势退化（v2 新增）

排查顺序：

1. 对比 `eval_pass_rate` 上一周期 vs 本周期，定位下降的 `case_class`。
2. 拉取 nightly eval 报告，找失败 case。
3. 对比代码 diff 与 prompt diff。
4. 必要时 revert 最近一次发布。

## 11. 成本控制

在线链路不建议每次都做重型验证。成本分层：

- **每请求必做**：request / node / retrieval count / citation count / token 统计 / cost 估算。
- **抽样做**：answer quality、faithfulness、claim-level verification。
- **离线做**：完整 eval suite、DPO / reward model 数据构建、策略对比。

建议增加 token / cost 预算：

| 场景          | 策略                                       |
| ------------- | ------------------------------------------ |
| definition    | 小上下文、小 token answer                  |
| comparison    | 增加 doc diversity，限制每文档 chunk 数    |
| multi-hop     | 限制 hop、subquery、total chunks           |
| tool workflow | tool result preview 化                     |
| long docs     | 先 retrieval，再 compression，不直接塞全文 |

成本一等指标 `llm_cost_estimated_usd_total{provider,model,route,price_version}` 必须按 route 在 dashboard §8.5 可视化，并配 P2 告警阈值。

## 12. 安全与隐私

监控系统本身也是潜在泄漏面，需要遵守数据最小化：

- metrics 不记录原始 query、原始 answer、完整 chunk。
- logs 只记录 preview 和脱敏 metadata，所有写入走 `redactor`。
- trace 默认不写敏感 tool output，必须开启 LangSmith 输入/输出 redactor。
- confirmation token 不进长期状态；自检任务 `security_token_leak_suspect_total` 守底。
- user_id 一律 hash 后再写 trace / log，metrics 只聚合 tenant / role。
- ACL denied、verifier block 只记录 reason code，不记录敏感目标内容。
- label 基数上限按 §4 硬性规则执行，禁止把 user_id / doc_id / 完整 query 直接打 label。

## 13. 阶段性验收指标

第一阶段上线后，应能回答以下问题：

- 当前服务 p99 是多少，最慢节点是谁？
- 一次错误回答是 top-k 没召回、rerank 误杀、citation 错，还是 LLM 没按证据回答？
- multi-hop 是真的覆盖了所有子问题，还是 fallback 到单跳？
- 工具调用是否经过 confirmation，是否发生重复执行？
- 当前 Chroma / SQLite / FTS5 是否一致，segment 是否异常膨胀？
- LLM 各 provider 限流、TTFT、成本分布是否健康？
- 某个版本发布后 route accuracy、citation、fallback、latency、eval pass rate 是否退化？

阶段性目标：

| 指标                                | 目标                                                        |
| ----------------------------------- | ----------------------------------------------------------- |
| `/health` 可用性                    | 99.9%+                                                      |
| `/chat` p95                         | MVP 可先 < 8s                                               |
| `/chat` p99                         | MVP 可先 < 20s                                              |
| LLM TTFT p95                        | < 1.5s（流式）                                              |
| citation refs valid                 | 99%+                                                        |
| knowledge health status             | 无持续 warning                                              |
| side-effect duplicate execution     | 0                                                           |
| eval pass rate                      | 不低于当前 baseline（multi-hop 子集 100%、全量 ≥ 现有水平） |
| Chroma segment anomaly              | 0                                                           |
| `embedding_dim_mismatch_total`      | 0                                                           |
| `security_token_leak_suspect_total` | 0                                                           |

## 14. 面试表达版本

可以这样介绍：

> 这个项目我没有只做"能回答"的 RAG，而是把它做成了一个可观测、可回归、可治理的企业 Agent 系统。监控分了七层：服务层看 QPS、错误率和 P99；Graph 层看 Supervisor、Planner、Tool、Verifier、Composer 各节点耗时和降级原因；RAG 层看 top-k、filtered、rerank、merged、citation 和 fallback；Multi-hop 层看 hop_count、per_subquery_coverage、global_coverage、decompose_error_code；Provider 层看 LLM/Embedding/Rerank 的限流、TTFT、token、估算成本、provider fallback；Tool 层看 confirmation、idempotency、timeout 和 verifier block；Knowledge 层看 SQLite、FTS5、Chroma 的一致性以及 Chroma segment size。
>
> 在 Phase 3 多跳 RAG 上，PR-8.4 baseline 的 `avg_hop_count` 从 0.75 提到 2.25，`avg_per_subquery_coverage` 从 0.50 提到 0.927，`avg_global_coverage` 0.925，`decompose_failed=0`，所以监控不是空喊"质量提升"，而是有可对齐的离线指标。我还把这些 eval 聚合值通过 nightly CI 推回在线 gauge，dashboard 上能直接看到 pass rate 的发布前后对比。
>
> 之前我还遇到过 Chroma eval 反复 delete/upsert 导致 HNSW segment 膨胀到 ~250GB 的问题，所以 health 不只看 chunk count，还会监控 segment size、平均每 chunk 占用和 content_hash skip rate；GLM embedding 也踩过 429，所以 LLM/embedding provider 都有独立监控、自动 fallback 和成本归因。整套体系把 tracing、metrics、eval、health、cost 连接起来，形成了 Agent 工程化闭环。

## 15. v1 → v2 变更摘要

- 新增 §3.5 LLM Provider 监控、§3.6 Embedding/Rerank Provider 监控、§3.10 Eval 持续监控。
- 新增 §4.1 在线 ↔ eval ↔ debug payload 三方字段映射；明确 histogram bucket 与 label 基数硬上限。
- 新增 §10.6 LLM provider 异常 runbook、§10.7 Eval 退化 runbook。
- 新增 PR-0（常量与契约）、PR-5（Provider Metrics）、PR-7（Eval 持续监控）；PR-1 删除"内存聚合 MVP"过渡态，明确多 worker 安全 backend；Knowledge Health Storage Metrics 升级为 P0 并前置为 PR-2。
- §3 内 histogram 数量类指标统一改名（`*_count` → `*_results`），避免与 Prometheus `_count` 后缀冲突。
- §6 trace、§5 日志统一走 `redactor`，硬性禁止全量 prompt/chunk/token 入 sink；新增 `observability_emit_failed_total` 自监控、`security_token_leak_suspect_total` 自检。
- §11 成本控制升级为一等指标（`llm_cost_estimated_usd_total`），配 P2 告警。
- §13 验收目标补充 LLM TTFT、embedding dim mismatch、token 泄漏自检三项硬指标。

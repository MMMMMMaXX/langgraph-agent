# Memory 模块说明

`app/memory/` 负责处理项目里的“会话后记忆”相关能力，但这里的 memory 不是单一概念，而是 3 层能力的组合：

1. **ordered history**：按时间顺序记录“用户问过什么”
2. **semantic memory**：为 follow-up / recall 提供语义召回
3. **memory node services**：为 `memory_node` 提供摘要刷新、写入、裁剪、debug 组装等内部服务

这份文档的目标是帮助你快速判断：修改某种行为时，应该进入哪个文件或哪个子目录，而不是把整个 `memory/` 当成一个“大而全模块”。

---

## 1. 目录职责总览

```text
app/memory/
├── history/              # 非向量化会话流水：有序历史、回放、总结用
├── services/             # memory_node 内部服务：summary / vector / history / prune / debug
├── conversation_history.py
├── vector_memory.py      # Chroma 语义记忆
├── write_policy.py       # 是否写 semantic memory 的统一决策
└── node_services.py      # 兼容门面，转发到 services/
```

可以把这几层理解成：

- `history/`：回答“之前问过什么”
- `vector_memory.py`：回答“之前聊过的相关内容能不能语义召回”
- `write_policy.py`：回答“这一轮值不值得进入 semantic memory”
- `services/`：回答“memory_node 这一轮怎么执行”

---

## 2. history：有序会话流水

相关文件：

- [history/service.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/history/service.py)
- [history/backend.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/history/backend.py)
- [history/sqlite_backend.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/history/sqlite_backend.py)
- [history/jsonl_backend.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/history/jsonl_backend.py)
- [history/events.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/history/events.py)
- [history/schema.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/history/schema.py)

### 它负责什么

这里保存的是**按时间顺序的会话事件**，核心用途是：

- `总结所有问题`
- `历史问题包括什么`
- `刚刚问了什么`

也就是说，history 是 **ordered replay source**，不是 graph checkpoint，也不是 semantic memory。

### 修改什么行为时应该改这里

适合改这里的需求：

- 改“总结历史”依赖的数据源
- 改 history 去重逻辑
- 改 SQLite / JSONL 后端
- 改 history event 的 schema
- 改请求级 history_path / backend 解析

不适合改这里的需求：

- semantic recall 质量问题
- memory 是否应该写入 Chroma
- message 裁剪策略

### 对外推荐入口

主链路和脚本都应该优先调用：

- `append_history_event(...)`
- `get_recent_history(...)`
- `get_all_history(...)`

而不是直接依赖具体 backend。

---

## 3. vector_memory：语义记忆层

相关文件：

- [vector_memory.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/vector_memory.py)

### 它负责什么

这里保存的是写入 Chroma 的**semantic memory**，核心用途是：

- follow-up 场景下的相关事实召回
- 长对话里的主题延续
- 语义相关而非严格有序的 recall

它不是顺序历史，也不是 runtime state 恢复源。

### 典型职责

- 生成 Chroma memory document id
- 构造 memory metadata
- 执行 add/upsert
- semantic query
- recent memory 窗口读取
- 关键词得分与 hybrid 排序辅助

### 修改什么行为时应该改这里

适合改这里的需求：

- memory 检索排序策略
- memory metadata schema
- Chroma 读写分页
- query embedding / memory embedding 的使用方式

不适合改这里的需求：

- 本轮是否应该写入 memory
- meta query 是否跳过写入
- history 总结行为

---

## 4. write_policy：是否写 semantic memory 的决策层

相关文件：

- [write_policy.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/write_policy.py)

### 它负责什么

这里专门负责一个问题：

**这一轮结果，值不值得写进 semantic memory？**

输出是 `MemoryWriteDecision`，里面包含：

- `should_write`
- `skip_reason`
- `memory_type`
- `tags`
- `memory_key`
- `confidence`
- `source_route`

### 为什么单独存在

因为“是否写 memory”是一个策略问题，不应该散落在 `memory_node` 或 `vector_memory.py` 里。

当前它会综合判断：

- route 类型
- 是否 tool request
- 是否 creative output
- 是否 RAG 定义类 doc hit
- answer 是否为空 / 太短 / 资料不足
- rewritten query 是否属于 meta query

### 修改什么行为时应该改这里

适合改这里的需求：

- 哪些路由默认不写 semantic memory
- 哪些回答太短 / 太差应该跳过
- memory_key / tags / memory_type 的生成策略

不适合改这里的需求：

- 实际怎么写 Chroma
- history 如何去重

---

## 5. services：memory_node 内部编排服务

相关目录：

- [services/__init__.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/services/__init__.py)
- [services/summary_service.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/services/summary_service.py)
- [services/vector_write_service.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/services/vector_write_service.py)
- [services/history_write_service.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/services/history_write_service.py)
- [services/prune_service.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/services/prune_service.py)
- [services/debug_payloads.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/services/debug_payloads.py)
- [services/types.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/services/types.py)

### 它负责什么

这一层是给 [app/nodes/memory.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/nodes/memory.py) 服务的内部编排模块。

职责分成 5 类：

1. `summary_service.py`
   - 是否刷新摘要
   - 调用 `summarize_messages`
   - 产出 `SummaryRefreshResult`

2. `vector_write_service.py`
   - 根据 `MemoryWriteDecision` 决定是否执行 Chroma 写入
   - 产出 `VectorStoreResult`

3. `history_write_service.py`
   - 决定是否写 history
   - 调用 `append_history_event`
   - 产出 `HistoryStoreResult`

4. `prune_service.py`
   - 裁剪 working memory 消息窗口
   - 产出 `MessagePruneResult`

5. `debug_payloads.py`
   - 组装 `debug_info[NODE_MEMORY]`
   - 组装 `log_node(extra=...)`

### 为什么这一层不是对外平台 API

这一层的定位是：

- **给 `memory_node` 用的内部服务**
- 不是 history/vector memory 的公共能力入口

也就是说：

- 主链路外部代码不应该直接依赖 `summary_service.py`
- 如果是“历史流水操作”，应该走 `history/service.py`
- 如果是“semantic memory 检索/写入”，应该走 `vector_memory.py`

---

## 6. memory_node：统一后处理编排层

相关文件：

- [app/nodes/memory.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/nodes/memory.py)

### 它负责什么

`memory_node` 是图里的后处理节点，负责把本轮回答的后续处理串起来：

1. summary 刷新
2. semantic memory 写入
3. ordered history 写入
4. working memory 裁剪
5. debug payload + structured log

### 它不负责什么

它不应该承担：

- 直接保存所有底层实现细节
- 直接成为 vector/history 的“平台 API”
- 继续膨胀成一个上帝函数

现在它的理想定位是：

- **只做 orchestration**

---

## 7. `conversation_history.py` 和 `node_services.py` 为什么还在

相关文件：

- [conversation_history.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/conversation_history.py)
- [node_services.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/node_services.py)

这两个文件当前主要是**兼容门面**：

- `conversation_history.py`
  - 兼容旧调用路径
  - 对外继续暴露 history service 高层接口

- `node_services.py`
  - 兼容旧导入路径
  - 实际实现已经迁到 `services/`

如果后续没有外部调用依赖，再考虑进一步弱化或删除。

---

## 8. 常见改动应该去哪里

### 场景 1：想让“总结所有问题”更准

优先看：

- [history/service.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/history/service.py)
- [app/agents/chat/history_summary.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/agents/chat/history_summary.py)

### 场景 2：想让跨轮 follow-up 召回更准

优先看：

- [vector_memory.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/vector_memory.py)
- [write_policy.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/write_policy.py)

### 场景 3：想调整哪些回答进入 semantic memory

优先看：

- [write_policy.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/write_policy.py)

### 场景 4：想让 memory_node 更薄、更清晰

优先看：

- [services/](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/services)
- [app/nodes/memory.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/nodes/memory.py)

### 场景 5：想改 SQLite / JSONL 的 history 持久化

优先看：

- [history/sqlite_backend.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/history/sqlite_backend.py)
- [history/jsonl_backend.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/history/jsonl_backend.py)
- [history/schema.py](/Users/manxin/baidu/ai-max/langgraph-agent/app/memory/history/schema.py)

---

## 9. 当前边界总结

一句话概括当前 `memory/` 的职责边界：

- `history/`：顺序历史
- `vector_memory.py`：语义记忆
- `write_policy.py`：是否写 semantic memory
- `services/`：memory node 内部步骤
- `memory_node`：统一编排

后续如果继续优化，优先遵守这个边界，而不是重新把逻辑塞回一个文件里。

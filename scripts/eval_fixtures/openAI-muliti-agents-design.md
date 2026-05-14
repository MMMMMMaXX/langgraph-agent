基于 OpenAI Agents SDK 的多 Agent 系统架构设计与演进实践

## 目录

1. [OpenAI Agents SDK 简介](#一openai-agents-sdk-简介)
2. [项目背景与问题](#二项目背景与问题)
3. [解决方案](#三解决方案)
4. [当前方案的局限性](#四当前方案的局限性)
5. [记忆系统设计原理](#五记忆系统设计原理)
6. [框架对比与选型](#六框架对比与选型)

---

## 一、OpenAI Agents SDK 简介

### 1.1 什么是 OpenAI Agents SDK

OpenAI Agents SDK 是 OpenAI 官方发布的轻量级 Python 库，用于构建 Agentic AI 应用。它是内部实验项目 [Swarm](https://github.com/openai/swarm) 的生产级演进版本，于 2025 年正式开源。

> GitHub: [https://github.com/openai/openai-agents-python](https://github.com/openai/openai-agents-python)
> 官方文档: [https://openai.github.io/openai-agents-python/](https://openai.github.io/openai-agents-python/)
> **两大设计原则**：

1. **够用但精简** — 功能足够实用，但原语足够少，学习曲线平缓
2. **开箱即用但可深度定制** — 默认配置即可工作，同时支持无限扩展

### 1.2 核心特性

**基础特性（2025 年 3 月 v0.0.2 起即支持）**

| 特性                  | 说明                                                |
| --------------------- | --------------------------------------------------- |
| **Agents**            | 核心构建块，LLM + instructions + tools 的组合       |
| **Handoffs**          | Agent 间任务委托，LLM 自主决定何时切换              |
| **Sessions**          | SQLiteSession，跨轮次自动管理对话历史               |
| **Tracing**           | 内置追踪，支持可视化、调试和优化工作流              |
| **Tools**             | function_tool 装饰器定义工具                        |
| **Output Types**      | 结构化输出，Pydantic 模型定义返回格式               |
| **Context**           | 依赖注入机制，在 Agent/Tool/Handoff 间共享状态      |
| **MCP Servers**       | 支持 MCP 协议，接入外部工具服务                     |
| **Guardrails**        | 输入输出安全校验，可配置的防护机制（tripwire 机制） |
| **Human in the loop** | 工具调用的 approval 暂停/恢复机制                   |

**近期更新特性** 🆕

| 特性                    | 版本    | 说明                                         |
| ----------------------- | ------- | -------------------------------------------- |
| **Realtime Agents**     | v0.3.0+ | 基于 `gpt-realtime-1.5` 构建低延迟语音 Agent |
| **Websocket Transport** | v0.10.0 | Responses API 的 websocket 传输支持          |
| **Any-LLM Support**     | v0.12.0 | 通过 LiteLLM 支持 100+ LLM 提供商            |
| **MongoDB Session**     | v0.13.0 | MongoDB 会话后端                             |
| **Sandbox Agents**      | v0.14.0 | 🆕 Agent 在隔离沙箱中操作文件/执行命令       |
| **Skills**              | v0.14.0 | 🆕 技能发现与按需加载（渐进式披露）          |
| **Sandbox Memory**      | v0.14.0 | 🆕 跨运行记忆，Agent 可从历史运行中学习      |
| **Workspace Mounts**    | v0.14.0 | 🆕 支持 S3/R2/GCS/Azure 远程存储挂载         |
| **Snapshot & Resume**   | v0.14.0 | 🆕 工作空间快照保存与运行状态恢复            |

> **关于 v0.14.0 Sandbox Agents**：SDK 新增了 `SandboxAgent` + `Manifest` + `SandboxRunConfig`，在隔离沙箱中提供 Shell / Filesystem / Skills / Memory / Compaction 五种内置能力，执行后端支持本地、Docker 和 7 家托管服务。这是 SDK 向"长任务 + 跨运行记忆"方向的重要演进，与本项目的记忆系统演进方向高度相关，有兴趣可以在 Q&A 展开。
> 📌 **版本里程碑**：
>
> - **2025-03**：v0.0.2 正式发布（首个稳定版本）
> - **v0.9.0**：要求 Python 3.10+
> - **v0.14.0**：引入 Sandbox Agents（重大更新）
> - **v0.14.2**：当前最新版本

### 1.3 核心原语

SDK 只有极少的核心原语，结合 Python 即可表达复杂的 Agent 关系：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      OpenAI Agents SDK 核心原语                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Agent                    Runner                  Session              │
│   ├─ name                  ├─ run() / run_sync()   ├─ get_items()       │
│   ├─ instructions          ├─ max_turns            ├─ add_items()       │
│   ├─ model                 └─ run_config           └─ 持久化存储         │
│   ├─ tools                                                              │
│   ├─ handoffs              Handoff                 Guardrail            │
│   ├─ output_type           ├─ agent                ├─ input validation  │
│   ├─ mcp_servers           ├─ on_handoff           └─ output validation │
│   └─ guardrails            └─ input_type                                │
│                                                                         │
│                            RunConfig                                    │
│                            ├─ model_settings                            │
│                            ├─ call_model_input_filter ← 模型调用前钩子  │
│                            └─ tracing_disabled        ← 追踪开关        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Agent 主要配置项**：

- `name`: Agent 名称（必填）
- `instructions`: 系统提示词（developer message）
- `model`: 使用的 LLM 模型
- `tools`: Agent 可用的工具列表
- `handoffs`: 可转移控制权的目标 Agent
- `output_type`: 结构化输出类型（Pydantic 模型）
- `input_guardrails` / `output_guardrails`: 输入输出校验规则

### 1.4 简单 Demo

**Demo 1：Hello World**

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")
result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

**Demo 2：多 Agent Handoff**

```python
from agents import Agent, Runner

breakdown_agent = Agent(
    name="拆分专家",
    instructions="将需求拆分为具体任务"
)

analyst = Agent(
    name="需求分析师",
    instructions="分析用户需求，分析完成后转交给拆分专家",
    handoffs=[breakdown_agent]  # 声明可以 handoff 给谁
)

result = Runner.run_sync(analyst, "我需要一个用户管理系统")
# LLM 自主决定何时调用 handoff，切换到 breakdown_agent
```

> 其他能力（`function_tool` 定义工具、`output_type` 结构化输出、`SQLiteSession` 持久化）在 §3 讲项目扩展时会结合使用。

### 1.5 Agent Loop 执行原理

SDK 的核心是 **Agent Loop** —— 一个持续运行直到获得最终输出的循环。它是整个 SDK 的"发动机"：你看到的 `Runner.run(agent, input, session)` 一行调用，内部就是这个循环在驱动。

#### 1.5.1 循环总览

```
输入 Prompt
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      AGENT LOOP                             │
│                                                             │
│  turn = 0                                                   │
│  while turn < max_turns:                                    │
│      turn += 1                                              │
│                                                             │
│      1. 组装模型输入（system + history + tools schema）     │
│      2. 调用 LLM                                            │
│      3. 解析响应：                                          │
│         ├─ Tool Call     → 执行工具 → 结果入历史 → 继续    │
│         ├─ Handoff       → 切换 current_agent → 继续       │
│         └─ Final Output  → 校验 output_type → 退出循环     │
│                                                             │
│      4. 将新产生的 items 写回 Session                       │
│                                                             │
│  超过 max_turns 未得到 Final Output → MaxTurnsExceeded      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
返回 Result(final_output, new_items, last_agent)
```

几个需要先明确的前提：

- **一次 **`Runner.run()`** 调用 = 一次 Agent Loop**；但 Loop 内部会**多次调用 LLM**——每轮都以一次 LLM 调用为界，LLM 让执行工具、或让切换 Agent，都会再触发下一轮 LLM 调用，直到 LLM 产出 Final Output
- **循环边界**：`max_turns`（默认 10）保护系统不因工具调用失控而无限转
- **当前 Agent（current_agent）是一个循环内变量**：Handoff 时它被替换，其 `instructions` 和 `tools` 也跟着换，但 **Session 和消息历史继续延续**

#### 1.5.2 每一轮内部的四个阶段

**阶段 A：组装模型输入**

SDK 把当前这一轮要发给 LLM 的消息列表拼出来，顺序固定：

```
┌──────────────────────────────────────────────────────────────┐
│  LLM 输入 = [                                                 │
│    1. system message   ← 来自 current_agent.instructions     │
│                          （支持动态函数：(ctx) -> str）       │
│    2. history messages ← session.get_items() 返回的历史       │
│    3. new input        ← 本次 Runner.run() 的 input 参数      │
│    4. tool results     ← 上一轮 Tool Call 的执行结果（如有）  │
│  ]                                                           │
│                                                              │
│  同时附带：                                                   │
│    • tools schema   ← current_agent.tools 每个工具的 JSON schema │
│    • handoffs schema ← 每个 handoff 自动生成 transfer_to_*    │
│    • output_type    ← 若定义，传入 response_format（JSON Schema）│
└──────────────────────────────────────────────────────────────┘
```

关键点：**handoff 对 LLM 来说也是一个工具**。SDK 自动为 `agent.handoffs` 里的每个目标生成一个名为 `transfer_to_<target_agent_name>` 的伪工具，LLM "调用"它就等于触发切换。

**阶段 B：调用 LLM**

带着上面拼好的消息和 schema，调用模型。`ModelSettings`（温度、top_p、tool_choice 等）在此生效。调用前后是 SDK 两个关键扩展点——详见 §1.5.4。

**阶段 C：解析响应（三选一分支）**

LLM 返回后，SDK 按下面的优先级判定走哪条分支：

| 判定条件                                            | 分支             | 后续动作                                           |
| --------------------------------------------------- | ---------------- | -------------------------------------------------- |
| 响应包含 `tool_calls`，且其中有 `transfer_to_*`     | **Handoff**      | 切换 `current_agent`，**保留 history**，进入下一轮 |
| 响应包含 `tool_calls`（非 handoff）                 | **Tool Call**    | 执行工具，结果追加为 `role=tool` 消息，进入下一轮  |
| 无 `tool_calls`，且输出符合 `output_type`           | **Final Output** | 退出循环                                           |
| 无 `tool_calls`，但没有 `output_type`（纯文本模式） | **Final Output** | 退出循环，`final_output` = 文本                    |

**阶段 D：写回 Session**

本轮新产生的 items（user input、assistant message、tool_calls、tool_results、handoff 记录）被 `session.add_items()` 追加持久化。**Session 是跨轮、跨 Runner.run() 调用**连续生长的。

#### 1.5.3 三种分支的执行细节

**① Tool Call 分支**

```
LLM 输出 tool_calls=[{name: "search", args: {...}}, {name: "fetch", args: {...}}]
   │
   ▼
SDK 按 tool_calls 列表【串行】执行（不是并行）
   │
   ├─ 执行 search(...) → 返回结果
   ├─ 执行 fetch(...)  → 返回结果
   │
   ▼
把结果作为 role=tool 消息追加到历史
   │
   ▼
进入下一轮 LLM 调用（LLM 看到工具结果后，决定下一步）
```

常见坑：

- **工具异常不会中断循环**：异常被转换成 `role=tool` 错误消息喂回去，LLM 可能"看到报错然后重试"或"换个参数调用"。这意味着一个工具反复抛错可能把 `max_turns` 耗光。

**② Handoff 分支**

```
LLM 调用 transfer_to_BreakdownAgent(input={...})
   │
   ▼
SDK 识别为 Handoff（不是普通工具）
   │
   ▼
触发 on_handoff 回调 → input_filter（可选，过滤/改写历史）
   │
   ▼
current_agent = BreakdownAgent
   │  （instructions / tools / handoffs 全部换成新 Agent 的）
   ▼
保留消息历史，进入下一轮
   │  （新 Agent 能看到之前的对话，但 system prompt 已换）
```

关键特性：

- **Handoff 是单向的**：A handoff 到 B 后，B 不会自动回到 A。如果需要回流，要显式把 A 加到 B 的 `handoffs` 列表
- **input_filter 的用处**：有时候你不想让 B 看到 A 的全部历史（比如 A 里有调试噪音），可以用 `handoff(..., input_filter=...)` 过滤
- **消息归属**：Handoff 后产生的所有消息仍在**同一个 Session**里，只是 `last_agent` 在变

**③ Final Output 分支**

```
无 tool_calls 的 LLM 响应：
   │
   ├─ 有 output_type → 按 JSON Schema 校验 + 解析为 Pydantic 对象
   │                   校验失败 → 继续循环（给 LLM 重试机会）
   │
   └─ 无 output_type → 文本直接作为 final_output 返回
```

退出后，Runner 返回 `RunResult`，包含：

- `final_output`：最终输出（字符串或 Pydantic 对象）
- `new_items`：本次 `run()` 产生的所有新消息（不含 run 之前就在 Session 里的历史）
- `last_agent`：最后活跃的 Agent（Handoff 后可能和传入的 agent 不是同一个）

#### 1.5.4 扩展点在循环中的位置

把循环当成时间轴，SDK 提供了这几个可插入的钩子：

```
一次 Agent Loop 内的完整时序
═══════════════════════════════════════════════════════════════

Runner.run(agent, input, session, run_config)
  │
  ├─ [Hook] on_agent_start(agent)                            ① 生命周期
  │
  ├─ session.get_items()                                     ② Session 读
  │
  ├─ [Hook] session_input_callback(history, new_input)       ③ 重塑上下文
  │           └─ 在 RunConfig 中注册；用于注入摘要、限制数量
  │
  ├─ [Guardrail] input_guardrails.check(user_input)          ④ 输入校验
  │           └─ tripwire → 中止，不调用 LLM
  │
  │  ┌──── while 循环 ────┐
  │  │                    │
  │  │  ├─ [Hook] call_model_input_filter(model_data)   ⑤ 调模型前最后一道
  │  │  │                                                  （监测 / 可选裁剪）
  │  │  ├─ call_model()                                   ⑥ LLM 调用
  │  │  │
  │  │  ├─ 响应解析：
  │  │  │   ├─ Tool Call:
  │  │  │   │    ├─ [Hook] on_tool_start(tool)          ⑦ 工具开始
  │  │  │   │    ├─ 执行工具
  │  │  │   │    └─ [Hook] on_tool_end(tool, result)    ⑧ 工具结束
  │  │  │   │
  │  │  │   ├─ Handoff:
  │  │  │   │    └─ [Hook] on_handoff(from, to)         ⑨ Handoff 切换
  │  │  │   │
  │  │  │   └─ Final Output:
  │  │  │        └─ [Guardrail] output_guardrails.check ⑩ 输出校验
  │  │  │                        tripwire → 阻断该输出
  │  │  │
  │  │  └─ session.add_items(new_items)                 ⑪ Session 写
  │  │
  │  └────────────────────┘
  │
  ├─ [Hook] on_agent_end(agent, result)                      ⑫ 生命周期
  │
  └─ return RunResult
```

本项目 **§3.5 Token 三层优化**正是挂在 ③（`session_input_callback`，重塑上下文结构）和 ⑤（`call_model_input_filter`，兜底监测）两个钩子上——这也是为什么本项目号称"不改 SDK 内核"的技术基础。

#### 1.5.5 一些容易忽略但重要的细节

- `instructions`** 支持动态生成**：可以传入 `(context, agent) -> str` 函数，每轮循环都会重新调用——意味着你可以在循环进行中"根据运行态改变 system prompt"
- **结构化输出的代价**：定义了 `output_type` 后，LLM 可能为了满足 JSON Schema 而"放弃"继续调用工具。遇到复杂任务建议仅在最后一个 Agent 上设 `output_type`
- **Tracing 自动埋点**：每个阶段 SDK 默认开启 `trace()` span，可在 OpenAI Dashboard 或自托管后端可视化。本项目因为自己有 metrics 体系，通过 `set_tracing_disabled(True)` 关闭了
- `max_turns`** 触发后**：抛 `MaxTurnsExceededError`，此前产生的消息**已经落库**——所以重试要考虑幂等性
- **Session 的"全量 vs 受限"**：SDK 原生 `session.get_items()` 默认返回全量历史；本项目通过继承 `AdvancedSQLiteSession` 重写为"按类型限定数量"，SDK 对此完全无感知（见 §3.2.6）

### 1.6 多 Agent 编排模式

官方文档推荐两种主要模式：

- **Manager（Agent-as-Tool）**：中心化控制。Manager Agent 保持对话主导权，将 SubAgent 作为 Tool 调用，上下文始终汇聚在 Manager 处。
- **Handoffs（对等转移）**：去中心化。Agent 之间通过 `handoff` 完全转移控制权，适合明确的专业分工场景。

本项目目前**两种原生模式都没有采用**，而是走**应用层编排**的第三条路线：由外层调度代码为每个 Agent 独立发起 `Runner.run()`、各自绑定不同类型的 Session，Agent 之间通过"跨 Session 声明式只读"共享信息（详见 §3.2.4 的说明与 §3.4）。这个选择换来了清晰的 Session 边界和细粒度权限，代价是放弃了 LLM 自主路由能力——对应 §4.1 "Agent 路由由应用层固定编排"局限。

---

## 二、项目背景与问题

### 2.1 项目简介

智能需求助手系统，基础架构图:

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=bc8c2391515b4f35855908823065a887&docGuid=06DKwVaenNtAcm)
如上图的 Agent 层,有不同角色的 Agent，但**不是扁平地并排摆着**——它体现了多 Agent 系统在真实业务里的两种常见形态：

- **依赖链**：上游 Agent 的产出是下游 Agent 的输入（分析 → 拆分 → 测试文档）
- **独立角色**：有些 Agent 完全自成闭环，不依赖别人，也不被别人直接依赖（事实收集、全局对话）

我们选择了 OpenAI Agent SDK 来做这套框架的底座. 经过实践,也踩了一些坑,回过头来我们看看问题及解决方案.

### 2.2 需要解决的四个核心问题

四个问题不是孤立的业务抱怨，而是在 SDK 原生原语上"照着文档写"之后会自然遇到的结构性缺口。下面按"现象 → 具体场景 → 技术根因 → 业务影响"的顺序展开。

---

#### 问题 1：上下文混乱（需要隔离）

**现象**：按 SDK 文档最直接的写法，多个 Agent 协作时共用同一个 Session；所有 Agent 的对话、工具调用、中间推理都追加到同一条消息流里。

**具体场景**：

```
一次需求分析流程：
  用户："我需要一个调研功能"
    │
    ▼
  AnalystAgent 分析 → 写入 session:
    • assistant: "正在调用 search_knowledge(...)"
    • tool: "搜索结果：..."（2000 字调研资料）
    • assistant: "分析完成，需要问卷设计+数据分析..."
    │
    ▼
  INFO_COLLECT 阶段写入 session:
    • 事实 1: "目标用户是 18-35 岁"
    • 事实 2: "预算 2 周"
    ...（10+ 条事实确认对话）
    │
    ▼
  BreakdownAgent 被调用时读到的 history：
    → 包含 AnalystAgent 的工具调用噪音（2000 字搜索结果）
    → 包含 INFO_COLLECT 的一来一回对话（20 条消息）
    → 真正需要的"分析结论"反而被淹没
```

**技术根因**：

- SDK 的 `session` 是 `Runner.run(agent, input, session=...)` 的入参，**Agent 本身不持有 Session**
- SDK Handoff 切换只换 `current_agent`，**Session 不切换**（见 §1.5.3）
- `session.get_items()` 默认返回**全量历史**，所有写入该 session 的 items 每次都会被读出来喂给下一个 Agent

**业务影响**：

- **Token 无谓消耗**：BreakdownAgent 根本不需要看搜索原文，但被强制喂进上下文
- **LLM 判断退化**："Lost in the Middle"——中间的拆分相关信息注意力被稀释
- **行为漂移**：BreakdownAgent 可能被 AnalystAgent 的工具调用格式"带偏"，自己也开始尝试调用不该调的工具

---

#### 问题 2：状态丢失（消息 ≠ 状态）

**现象**：Session 只有"消息序列"这一种存储形态；真正结构化的业务结论被迫以对话的形式记录，和消息一起被摘要/裁剪。

**具体场景**：两类信息在生命周期上的本质差异

```
┌─────────────────────────┬──────────────────────────────────────┐
│ 消息型数据（可裁剪）      │ 状态型数据（不能丢）                   │
├─────────────────────────┼──────────────────────────────────────┤
│ "我需要调研功能"         │ requirement.document = "..."          │
│ "能做问卷吗"             │ requirement.breakdown = [             │
│ "可以，我们拆成..."       │     {task: "问卷设计", priority: 1}, │
│ "再加数据分析"           │     {task: "数据分析", priority: 2}  │
│ （可以被摘要成一句话）    │ ]                                     │
│                         │ （这是最新结论，必须精确保留）          │
└─────────────────────────┴──────────────────────────────────────┘
```

如果把 `requirement.breakdown` 也放进消息流：

```python
session.add_items([{"role": "assistant", "content":
    "拆分结果：[{'task': '问卷设计', ...}, ...]"}])
```

那么第 30 轮之后（触发第 1 层消息数量限制），这条消息**连同下游 Agent 依赖的拆分结论一起消失**。BreakdownAgent 下一次被调用时会发现"之前明明拆过了，现在看不到了"。

**技术根因**：

- SDK `Session` 协议只规定 `get_items` / `add_items` / `pop_item` / `clear_session` 四个方法，**没有 key-value 状态槽位**
- 任何为了压缩上下文做的裁剪（按条数、按 Token 上限、按摘要替换）都会把业务结论一起裁掉
- SDK 原生不提供"版本"语义——同一个结论被多次更新时，早期版本无法追溯

**业务影响**：

- **结论回滚**：下游 Agent 读不到上游的定版结论，需要用户重复说或重新推理
- **多 Agent 协作断链**：Agent A 产出的 `analysis.result` 被裁剪后，Agent B 拿不到，退回"从零开始"
- **审计缺失**：某条分析结论在 10:15 是 A、在 10:30 变成了 B，过程追溯不回去

---

#### 问题 3：信息孤岛与权限控制

**现象**：做了 Session 隔离之后出现反向问题——Session 之间完全不通，Agent 看不到别的 Agent 的产出；但完全打开又等于没做隔离。

**具体场景**：需求拆分 Agent 需要看分析结论

```
用户在 FEATURE_CONTEXT 中完成需求分析：
  analysis.result = "用户调研 + 问卷设计 + 数据可视化"
  ────────────────────────────────────────
  写入 session_key =
  "alice@x.com_feature_F001_session"

用户在 INFO_COLLECT 中补充了 15 条事实：
  facts = [{目标用户: ...}, {预算: ...}, ...]
  ────────────────────────────────────────
  写入 session_key =
  "alice@x.com_info_collect_F001_session"

BreakdownAgent 启动时传入 session：
  Runner.run(breakdown_agent, input, session=???)
                                            ↑
                    这里只能传一个 session
                    传 FEATURE_CONTEXT → 看不到 INFO_COLLECT 的事实
                    传 INFO_COLLECT → 看不到 analysis.result
```

用户视角的失败路径：

```
用户："基于刚才的分析和补充，帮我拆一下"
BreakdownAgent："请问是什么需求？"
用户："我刚才不是说过了吗？！"
```

**另一个反向风险**：如果粗暴让 BreakdownAgent 读所有 session：

- 泄漏 `CONVERSATION_CONTEXT` 中的全局用户偏好（与当前 feature 无关）
- 读到其他 feature 的 `feature_F099_session`（属于别的任务，可能还是别人的）
- 读到 `INFO_COLLECT` 中敏感字段如 `secret.credentials`（如果未来有）

**技术根因**：

- SDK `Session` 实例由 `session_id` 唯一标识，**不同 session_id 之间没有任何 API 通路**
- Session 接口没有"只读打开 / 权限声明"这类原语，每个 Session 要么独占写，要么不存在于当前 `Runner.run()` 的视野
- 想共享只能暴力 merge 两个 session 的 items，但合并后既无隔离也无权限可言

**业务影响**：

- **用户体验崩坏**：重复表述、"我说过了"
- **安全面扩大**：任何 Agent 访问任何 Session 等于没有边界
- **无法演进**：未来多租户 / 多 feature 并行时，没有权限声明无法审计和收敛

---

#### 问题 4：Token 爆炸

**现象**：对话轮次增长后，单次 `Runner.run()` 喂给 LLM 的 Token 线性甚至超线性增长，直到撞到上下文窗口上限或成本不可控。

**具体场景**：一次典型的技术需求分析流程的 Token 增长（实际业务数据的估算）

```
阶段                               本轮消息     累计 history Token  单次 run 输入
──────────────────────────────────────────────────────────────────────
T0  用户首次提问                    1 user        ~50                ~2K (+system/tools)
T3  初步澄清 3 轮                  7 items       ~800               ~3K
T8  补充 5 个事实确认              17 items      ~2.5K              ~5K
T15 首次调用检索工具（返回3000字）  21 items      ~5K                ~7K
T25 基于检索迭代讨论 10 轮          41 items      ~9K                ~11K
T40 生成初版分析文档                45 items      ~13K               ~15K
T60 修订 + 再次检索                 65 items      ~22K               ~24K ⚠ 接近限制
T100 完整需求+拆分+测试文档         120+ items    ~40K+              ~42K+ ✗ 超限
```

注意这只是**单个 Agent、单个 Session**的增长。多个 Agent 共用同一 Session 时（问题 1 的场景），叠加速度更快。

**技术根因**：

- `SQLiteSession.get_items()` **默认返回全量**历史（源码 `SELECT * FROM agent_messages WHERE session_id = ?`）
- 每一轮 Agent Loop 在组装模型输入时都会把**整段历史重发一次**
- tool_call 模式下每次工具调用会追加 2 条 item（assistant{tool_calls} + tool{result}），工具结果可能是整段检索出来的原文，单条就数千 Token
- 没有原生摘要机制——SDK 把"怎么压缩上下文"完全留给应用层

**业务影响**：

- **窗口上限**：主流模型 8K / 32K / 128K 窗口都会在长流程中被逼近；超过后 API 直接报错
- **成本线性放大**：输入 Token 计费 × 轮数——100 轮相对 10 轮，累计输入 Token 可能涨 **40–80 倍**（每轮都带全量）
- **语义退化**：上下文过长时 LLM 注意力分散，关键指令和早期结论被模型"忘掉"，回答质量下降
- **延迟增加**：单次请求的 Prompt 处理时间随 Token 线性上升

---

**四个问题之间的结构关系**：

```
问题 1 (上下文混乱)  ──解决方式: 按类型隔离──▶  问题 3 (信息孤岛)
      ▲                                              │
      │                                              ▼
      │                                   解决方式: 声明式跨 Session 只读
      │
问题 2 (状态丢失) ──独立机制: 版本化 State 存储
问题 4 (Token 爆炸) ──独立机制: 三层优化架构
```

问题 1 的解法会诱发问题 3；问题 2 和问题 4 相对独立。这也是 §3 解决方案按四个子章节展开的原因。

---

## 三、解决方案

### 3.1 整体架构

整体方案的核心思路是一句话：**不改 SDK 内核，只在 Session 与 Runner 的外围补齐四件事——上下文隔离 / 跨 Session 声明式共享 / 版本化状态 / Token 三层优化**。

落到工程上，上面四件事被组织到三层组件里：

- **FlowRunner**（系统级封装层）：包住原生 `Runner.run()`，负责把"权限声明、上下文注入、RunConfig 钩子"这些横切能力在每次调用前组织好
- **FlowCortexSession + SessionState**（存储扩展层）：继承 SDK 的 `AdvancedSQLiteSession`，在消息存储之外独立挂一条版本化 State 存储线
- **Token Filter + Summary Callback**（RunConfig 钩子层）：通过 SDK 官方扩展点挂进 Agent Loop，不侵入 SDK 代码

```
┌─────────────────────────────────────────────────────────────────────┐
│                      项目架构概览                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                           API 请求                                  │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                     FlowRunner（系统级封装）                   │  │
│   │                                                               │  │
│   │  1. 读取 Agent 的跨 Session 权限声明                          │  │
│   │  2. 构建跨 Session 上下文块                                    │  │
│   │  3. 注入到 Prompt                                              │  │
│   │  4. 创建 Token Filter（RunConfig hooks）                       │  │
│   │  5. 调用 SDK Runner.run()                                      │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                   SDK Runner.run()                            │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│           ┌──────────────────┼──────────────────┐                   │
│           ▼                  ▼                  ▼                   │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│   │Extended      │   │ SessionState │   │ Token Filter │           │
│   │Session       │   │ (版本化存储) │   │ (调用前监测) │           │
│   │(限量+增量摘要)│  │              │   │              │           │
│   └──────────────┘   └──────────────┘   └──────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**各组件的角色与对应关系**：

| 组件                                                       | 层次           | 主要职责                                               | 解决的问题                     |
| ---------------------------------------------------------- | -------------- | ------------------------------------------------------ | ------------------------------ |
| **FlowRunner**                                             | 封装层         | 横切能力的组织点：权限读取、上下文注入、RunConfig 挂钩 | 所有问题的统一入口             |
| **Extended Session**（FlowCortexSession + SessionManager） | 存储层         | 类型化 Session 体系 + 实例缓存 + 增量摘要              | 问题 1（隔离）+ 问题 4 第 2 层 |
| **SessionState**                                           | 存储层         | 独立于消息的版本化 key-value 存储                      | 问题 2（状态丢失）             |
| **跨 Session Middleware**                                  | 运行时         | 按 Agent 声明的权限，以只读模式加载其他 Session        | 问题 3（信息孤岛）             |
| **Token Filter + Summary Callback**                        | RunConfig 钩子 | 调用 LLM 前重塑上下文 / 兜底监测                       | 问题 4（Token 爆炸）           |

**一次请求的执行时序**（对应 §1.5.4 的钩子时序）：

```
API 请求进入
    │
    ▼
FlowRunner.run(agent, prompt, email, ...)
    │
    ├─ [1] SessionManager 拿主 Session                 ← §3.2
    │
    ├─ [2] 读取 agent.cross_session_read_policy()       ← §3.4
    │     读取 agent.cross_session_state_read_policy()
    │
    ├─ [3] CrossSessionContextMiddleware 以只读模式      ← §3.4
    │     加载被声明允许读取的其他 Session，拼成上下文块
    │
    ├─ [4] 把上下文块 prepend 到用户 prompt              ← §3.4
    │
    ├─ [5] 创建 RunConfig：                              ← §3.5
    │     • session_input_callback = summary_callback  （第 2 层）
    │     • call_model_input_filter = token_filter     （第 3 层）
    │
    ├─ [6] 交给 SDK Runner.run(agent, prompt,
    │                           session, run_config)
    │      │
    │      └─ 进入 Agent Loop（§1.5）：
    │          • 读 Session（第 1 层消息数限制生效）    ← §3.5
    │          • session_input_callback 重塑上下文     ← §3.5
    │          • call_model_input_filter 兜底监测      ← §3.5
    │          • 调用 LLM / 执行 tool / handoff...
    │          • 写回 Session
    │
    └─ [7] 返回 RunResult
```

核心设计原则有三条：

1. **SDK 零侵入**：所有扩展都挂在 `Session` 继承和 `RunConfig` 官方钩子上，SDK 升级不影响应用层——这是 §6 说"本项目是应用层扩展，不是新框架"的技术基础
2. **单向数据流**：消息只追加不覆盖；状态只追加新版本不覆盖旧版本；跨 Session 读取永远是只读——没有双向同步这种高复杂度的东西
3. **策略声明优于命令**：Agent 用 `cross_session_read_policy()` 这样的声明式接口表达意图，由框架在运行时解释执行；这样未来做智能路由、多租户扩展时，改的是策略解释器，不是每个 Agent 的业务代码

接下来四个小节（§3.2–§3.5）按 §2.2 的四个问题顺序，把每一块的设计细节展开。

### 3.2 解决方案 1：Session 存储设计（基础设施）

> **对应 §2.2 问题 1（上下文混乱）**：通过 Session 类型体系 + SessionManager 单例，让不同职责的 Agent 落到不同 Session，实现多 Agent 上下文物理隔离。

#### 3.2.1 SDK Session 原生设计

**存储实质**：SDK 使用数据库存储对话历史，通过 `session_id` 区分不同对话域。

```
┌─────────────────────────────────────────────────────────────────────┐
│                  SDK Session 存储结构（SQLite 为例）                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   数据库表结构：                                                      │
│   ─────────────                                                     │
│                                                                     │
│   agent_sessions 表（会话元数据）                                     │
│   ┌──────────────┬──────────────┬──────────────┐                    │
│   │ session_id   │ created_at   │ updated_at   │                    │
│   │ (PRIMARY KEY)│              │              │                    │
│   ├──────────────┼──────────────┼──────────────┤                    │
│   │ "user_123"   │ 2026-04-19   │ 2026-04-19   │                    │
│   │ "user_456"   │ 2026-04-18   │ 2026-04-19   │                    │
│   └──────────────┴──────────────┴──────────────┘                    │
│                                                                     │
│   agent_messages 表（消息数据）                                       │
│   ┌────┬──────────────┬─────────────────────────────────────────┐  │
│   │ id │ session_id   │ message_data (JSON)                     │  │
│   ├────┼──────────────┼─────────────────────────────────────────┤  │
│   │ 1  │ "user_123"   │ {"role":"user","content":"Hello"}       │  │
│   │ 2  │ "user_123"   │ {"role":"assistant","content":"Hi!"}    │  │
│   │ 3  │ "user_456"   │ {"role":"user","content":"Help me"}     │  │
│   └────┴──────────────┴─────────────────────────────────────────┘  │
│                                                                     │
│   session_id 的作用：                                                │
│   • 唯一标识一个对话/用户                                            │
│   • 查询时 WHERE session_id = ? 获取该对话的所有消息                 │
│   • 不同 session_id 之间完全隔离                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Runner 与 Session 的交互流程**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Runner.run() 与 Session 交互                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Runner.run(agent, "用户输入", session=session)                     │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │ Step 1: 获取历史                                              │  │
│   │ history = session.get_items(limit=None)  ← 默认获取【全量】   │  │
│   │                                                               │  │
│   │ SQL: SELECT message_data FROM agent_messages                  │  │
│   │      WHERE session_id = ?                                     │  │
│   │      ORDER BY id ASC                                          │  │
│   └─────────────────────────────────────────────────────────────┘  │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │ Step 2: 组装上下文                                            │  │
│   │ model_input = history + new_input  ← 全部历史 + 新输入        │  │
│   └─────────────────────────────────────────────────────────────┘  │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │ Step 3: 调用 LLM                                              │  │
│   │ response = model.call(model_input)                           │  │
│   └─────────────────────────────────────────────────────────────┘  │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │ Step 4: 存储新消息                                            │  │
│   │ session.add_items(new_items)                                 │  │
│   │                                                               │  │
│   │ SQL: INSERT INTO agent_messages (session_id, message_data)   │  │
│   │      VALUES (?, ?)                                            │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**SDK 原生能力与局限**：

| 能力                 | SDK 原生支持 | 说明                                                                                                                    |
| -------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------- |
| 消息持久化           | ✅           | `SQLiteSession` / `AdvancedSQLiteSession` 原生；Redis/MongoDB 等由扩展包或社区提供                                      |
| session_id 隔离      | ✅           | 同一 DB 中不同 `session_id` 的消息互不可见（SQL `WHERE session_id=?`）                                                  |
| 限制获取数量         | ⚠️ 部分      | 仅 `get_items(limit=N)` 方法级参数；**默认 **`limit=None`** 返回全量**，Runner 调用时不传 limit，没有"全局默认上限"配置 |
| **Session 类型体系** | ❌           | 只有一种 Session，无业务语义区分（临时/功能/对话/信息收集）                                                             |
| **SessionManager**   | ❌           | 无统一注册与缓存机制，每次都需自行实例化                                                                                |
| **跨 Session 读取**  | ❌           | Session 之间完全隔离，无内置跨会话读取 API                                                                              |
| **版本化状态**       | ❌           | 仅持久化消息历史，无独立的 key-value 状态存储，更无版本链                                                               |

#### 3.2.2 项目扩展设计

**针对 §3.2.1 的四项 SDK 缺口，本项目的补齐方案**（与 §2.2 的业务问题编号区分开——此处是"SDK 能力缺口"，不是"业务问题"）：

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SDK 原生 vs 项目扩展                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   缺口 A：无 Session 类型体系                                         │
│   ─────────────────────────                                         │
│   SDK 原生：所有 Agent 共用一个 Session → 对应 §2.2 问题 1           │
│   扩展方案：SessionType 枚举 + 按类型路由 Agent → §3.2.3              │
│                                                                     │
│   缺口 B：Session 实例无统一管理                                      │
│   ─────────────────────────                                         │
│   SDK 原生：SQLiteSession("id") 每次创建新实例                       │
│   扩展方案：SessionManager 单例 + 缓存池 → §3.2.5                     │
│                                                                     │
│   缺口 C：Session 之间无共享通道                                      │
│   ─────────────────────────                                         │
│   SDK 原生：Session 完全隔离 → 对应 §2.2 问题 3                      │
│   扩展方案：声明式跨 Session 只读 → §3.4                              │
│                                                                     │
│   缺口 D：只有消息、无状态存储                                        │
│   ─────────────────────────                                         │
│   SDK 原生：Session 协议无 kv 槽位 → 对应 §2.2 问题 2                │
│   扩展方案：SessionState 版本化存储 → §3.3                            │
│                                                                     │
│   （§2.2 问题 4 "Token 爆炸" 不在 Session 存储层解决，见 §3.5）      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3.2.3 Session 类型体系（自研扩展）

**核心思想**：不同 Agent、不同业务场景使用不同类型的 Session，实现上下文隔离。

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Session 类型体系                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   类型                   作用域                存储            存活周期         典型用途          │
│   ─────────────────────────────────────────────────────────────────────────────────────────  │
│   IN_MEMORY              user                  SQLite :memory: 进程生命周期    临时计算、一次性任务 │
│   FEATURE_CONTEXT        user + feature        SQLite 文件     长期持久化      需求分析、技术方案   │
│   CONVERSATION_CONTEXT   user + conversation   SQLite 文件     长期持久化      用户对话、全局偏好   │
│   INFO_COLLECT           user + feature        SQLite 文件     长期持久化      事实清单收集        │
│                                                                     │
│   Session ID 构造规则（按类型不同，扩展 SDK 的单一 session_id）：      │
│   ───────────────────────────────────────────────                   │
│   SDK 原生：  "user_123"                                             │
│   扩展后：                                                           │
│   • IN_MEMORY          → "{email}_temp_session"                     │
│   • FEATURE_CONTEXT    → "{email}_feature_{feature_id}_session"     │
│   • CONVERSATION_CONTEXT → "{email}_chat_{conversation_id}_session" │
│   • INFO_COLLECT       → "{email}_info_collect_{feature_id}_session"│
│                                                                     │
│   示例：                                                             │
│   • alice@example.com_feature_F001_session                          │
│   • alice@example.com_info_collect_F001_session                     │
│   • alice@example.com_chat_CONV123_session                          │
│                                                                     │
│   隔离保证：                                                         │
│   • 类型不同 → session_id 前缀不同 → 完全隔离                        │
│   • 用户不同 → session_id 中间段不同 → 完全隔离                      │
│   • Feature 不同 → session_id 后缀不同 → 完全隔离                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3.2.4 Agent 与 Session 的对应关系

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Agent 与 Session 类型对应关系                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Agent                    写入 Session           读取 Session       │
│   ─────────────────────────────────────────────────────────────     │
│   AnalystAgent             FEATURE_CONTEXT        INFO_COLLECT      │
│   (需求分析)               (分析结果)              (事实清单)        │
│                                                                     │
│   信息收集逻辑             INFO_COLLECT           -                 │
│   (写入 INFO_COLLECT)      (收集的事实)                              │
│                                                                     │
│   BreakdownAgent           FEATURE_CONTEXT        FEATURE_CONTEXT   │
│   (需求拆分)               (拆分结果)              + INFO_COLLECT    │
│                                                                     │
│   TestDocGeneratorAgent    FEATURE_CONTEXT        FEATURE_CONTEXT   │
│   (测试文档生成)           (测试文档)                                │
│                                                                     │
│   设计原则：                                                         │
│   • 每个 Agent 有明确的"主 Session"（写入目标）                      │
│   • Agent 可以读取其他 Session（需声明权限）                         │
│   • 同一 Agent 的多次调用 → 写入同一 Session（上下文连续）           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> **关于 Handoff 与 Session 绑定的说明**
> SDK 原生 Handoff 有一个硬约束：**一次 **`Runner.run()`** 内所有 Agent 共用同一个 Session**。
> 因为 `session` 是 `Runner.run(agent, input, session=...)` 的入参，Handoff 切换的只是
> `current_agent`（instructions / tools 换了），Session 不会跟着切换。
> 所以本项目中"不同 Agent 对应不同 Session"的设计，**不是通过 SDK Handoff 实现的**，
> 而是通过**应用层编排**：每个 Agent 独立 `Runner.run()`，各自传入自己主 Session；Agent 之间
> 的信息流动通过 §3.4 的"跨 Session 声明式只读"完成。
> 这是一个主动的取舍：**牺牲 SDK Handoff 的 LLM 自主路由能力，换取清晰的 Session 边界
> 和细粒度权限控制**。这也是 §4.1 列出"Agent 路由硬编码"作为当前局限的原因。

#### 3.2.5 SessionManager 设计（自研扩展）

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SessionManager 设计                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   核心职责：全局单例，管理所有 Session 实例                          │
│                                                                     │
│   class SessionManager:                                             │
│       _instance: SessionManager          # 单例                     │
│       _sessions: Dict[str, Session]      # Session 缓存池           │
│       _lock: Lock                        # 并发安全                  │
│                                                                     │
│   关键方法：                                                         │
│   ─────────────                                                     │
│   get_session(type, email, feature_id) → Session                    │
│       1. 构造 session_key = f"{type}_{email}_{feature_id}"          │
│       2. 如果 key 已存在 → 返回缓存的 Session                        │
│       3. 如果 key 不存在 → 创建新 Session，加入缓存                  │
│       → 保证：相同参数永远返回同一实例                               │
│                                                                     │
│   get_or_load_readonly_by_key(key) → Session                       │
│       → 返回只读 Session（用于跨 Session 读取）                      │
│       → create_tables=False，避免副作用                              │
│                                                                     │
│   close_session(key) / close_all_sessions()                         │
│       → 清理资源                                                     │
│                                                                     │
│   流程示意：                                                         │
│   ─────────────                                                     │
│   Agent 请求 Session                                                 │
│        │                                                            │
│        ▼                                                            │
│   SessionManager.get_session(FEATURE_CONTEXT, "alice@x.com", "F001")│
│        │                                                            │
│        ├─ 已存在？ ─Yes─▶ 返回缓存的 Session                        │
│        │                                                            │
│        └─ 不存在？ ─────▶ 创建 FlowCortexSession                    │
│                          │                                          │
│                          ├─ db_path: ./data/alice@x.com/F001.db     │
│                          ├─ session_id: feature_context_alice_F001  │
│                          └─ 加入缓存池 → 返回                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3.2.6 FlowCortexSession 扩展

```
┌─────────────────────────────────────────────────────────────────────┐
│                  FlowCortexSession（继承 SDK AdvancedSQLiteSession）│
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   class FlowCortexSession(AdvancedSQLiteSession):                           │
│       session_type: SessionType          # 类型标识                  │
│       _summary_timeline: List[Dict]      # 增量摘要时间线            │
│       _global_overview: str              # 全局概览                   │
│       _state_store: SessionStateStore    # 版本化状态存储            │
│       _state_cache: Dict[str, SessionState]  # 状态内存缓存          │
│                                                                     │
│   重写方法：                                                         │
│   ─────────────                                                     │
│   get_items() → List[Message]                                       │
│       # SDK 默认：返回全部历史                                       │
│       # 我们重写：根据配置返回限定数量（默认 20 条）                  │
│       # 可通过环境变量 SESSION_HISTORY_DEFAULT_LIMIT 配置            │
│       return self._get_recent_items(limit)                          │
│                                                                     │
│   新增方法：                                                         │
│   ─────────────                                                     │
│   get_summary() → str              # 获取增量摘要（注入到 Prompt）   │
│   update_summary(messages)         # 更新增量摘要                    │
│   get_state(key) → Any             # 获取状态（最新版本）            │
│   set_state(key, value, editor)    # 设置状态（追加新版本）          │
│   list_state_versions(key) → List  # 获取状态历史版本                │
│                                                                     │
│   与 SDK 的关系：                                                    │
│   ─────────────                                                     │
│   SDK Runner.run() 调用 session.get_items()                         │
│                         ↓                                           │
│   FlowCortexSession.get_items() 返回【受限数量】的历史              │
│                         ↓                                           │
│   SDK 无感知，但实际传递给 LLM 的上下文已被控制                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 解决方案 2：版本化状态存储

> **对应 §2.2 问题 2（状态丢失）**：在 Session 的消息存储之外，独立开一个 key-value + 版本的 State 存储；追加式写入，不随消息裁剪而丢失。

```
┌─────────────────────────────────────────────────────────────────────┐
│                      版本化状态存储                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   核心问题：Session 存储【对话历史】，但有些信息是【状态】而非对话   │
│   ─────────────────────────────────────────────────────────────     │
│   • 对话历史：消息序列，先后顺序重要，可以被摘要/裁剪               │
│   • 状态信息：结构化数据，是最新结论，不能丢失                      │
│                                                                     │
│   举例：                                                             │
│   • "用户需要登录功能" → 这是对话消息                               │
│   • requirement.document = "登录功能规格..." → 这是状态（分析结论）  │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   设计原则：状态与消息分离 + 追加式写入（不覆盖）+ 审计追踪          │
│                                                                     │
│   数据结构：                                                         │
│   ─────────────                                                     │
│   SessionState {                                                    │
│       key: str          // 状态键（如 "requirement.document"）      │
│       value: Any        // 状态值（JSON 可序列化）                  │
│       version: int      // 版本号（递增）                           │
│       editor: str       // 编辑者（如 "AnalystAgent"）              │
│       timestamp: float  // 时间戳                                   │
│   }                                                                 │
│                                                                     │
│   版本演进示例：                                                     │
│   ─────────────                                                     │
│   Key: "requirement.document"                                       │
│                                                                     │
│   V1 (AnalystAgent, 10:00) → "用户需要调研功能"                     │
│       ↓                                                             │
│   V2 (AnalystAgent, 10:15) → "用户需要调研功能，包含问卷设计"       │
│       ↓                                                             │
│   V3 (BreakdownAgent, 10:30) → "...问卷设计和数据分析"              │
│                                                                     │
│   API：                                                              │
│   • get_state(key) → 返回最新版本                                   │
│   • list_state_versions(key) → 返回所有历史版本                     │
│   • set_state(key, value, editor) → 追加新版本                      │
│                                                                     │
│   收益：                                                             │
│   • 状态不会因消息清理而丢失                                         │
│   • 支持历史回溯和审计                                               │
│   • 多 Agent 协作时可追踪谁修改了什么                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.4 解决方案 3：跨 Session 共享与权限控制

> **对应 §2.2 问题 3（信息孤岛与权限控制）**：在 §3.2 Session 隔离的基础上，通过声明式只读策略 + 最小权限原则，让 Agent 能看到它需要的、且只能看到它声明过的跨 Session 信息。

```
┌─────────────────────────────────────────────────────────────────────┐
│                  跨 Session 共享与权限控制                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   核心矛盾：完全隔离 → 信息孤岛；完全开放 → 权限失控                 │
│   解决方案：声明式跨 Session 读取 + 最小权限原则                     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Agent 权限声明                                                     │
│   ─────────────                                                     │
│   class BreakdownAgent(Agent):                                      │
│                                                                     │
│       def cross_session_read_policy(self):                          │
│           return [FEATURE_CONTEXT, INFO_COLLECT]                    │
│           # 声明可读取的 Session 类型                                │
│                                                                     │
│       def cross_session_state_read_policy(self):                    │
│           return CrossSessionStatePolicy(                           │
│               readable_states={                                     │
│                   FEATURE_CONTEXT: {"analysis.result"},             │
│                   INFO_COLLECT: {"requirement.document"}            │
│               }                                                     │
│           )                                                         │
│           # 声明可读取的具体 State Keys                              │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   跨 Session 共享流程                                                │
│   ─────────────────                                                 │
│                                                                     │
│   BreakdownAgent 启动                                               │
│         │                                                           │
│         ▼                                                           │
│   FlowRunner 检查权限声明                                           │
│   → cross_session_read_policy(): [FEATURE_CONTEXT, INFO_COLLECT]   │
│         │                                                           │
│         ▼                                                           │
│   CrossSessionContextMiddleware                                     │
│   → 以只读模式获取 FEATURE_CONTEXT Session                          │
│   → 以只读模式获取 INFO_COLLECT Session                             │
│         │                                                           │
│         ▼                                                           │
│   构建上下文块 → 注入 Prompt → 调用 SDK Runner                      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   权限检查示例                                                       │
│   ─────────────                                                     │
│   请求：BreakdownAgent 读取 INFO_COLLECT 的 "secret.data"           │
│                                                                     │
│   检查 1: Session 类型是否允许？                                     │
│   → INFO_COLLECT ∈ [FEATURE_CONTEXT, INFO_COLLECT] → ✅             │
│                                                                     │
│   检查 2: State Key 是否允许？                                       │
│   → "secret.data" ∈ {"requirement.document"} → ❌ 拒绝访问          │
│                                                                     │
│   结果：未声明的资源无法访问（最小权限原则）                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.5 解决方案 4：Token 优化三层架构

> **对应 §2.2 问题 4（Token 爆炸）**：三层递进——第 1 层限数量、第 2 层做增量摘要、第 3 层兜底监测——从消息数、信息密度、调用前校验三个维度把 Token 增长曲线从"线性发散"压到"稳态收敛"。

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Token 优化三层架构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   第 1 层：消息数量限制                                              │
│   ─────────────────────                                             │
│   原理：重写 Session.get_items()，限制返回的历史消息数量             │
│   配置：默认 20 条，可通过 SESSION_HISTORY_DEFAULT_LIMIT 配置        │
│   效果：50轮 25K → 10K (减少 60%)                                   │
│                                                                     │
│                              ↓                                      │
│                                                                     │
│   第 2 层：增量摘要                                                  │
│   ─────────────────────                                             │
│   原理：将早期历史压缩为摘要，只保留关键信息                         │
│   策略：每 5 轮提取【新增信息】追加到 timeline                       │
│         每 10 个增量压缩生成 global_overview                        │
│         （5 / 10 为代码常量，当前未暴露为配置项）                     │
│   构成：global_overview(~200) + 最近8增量(~400) + 最近10消息(~2000) │
│   效果：100轮 10K → 6K (再减少 40%)                                 │
│                                                                     │
│                              ↓                                      │
│                                                                     │
│   第 3 层：Token 监测（兜底裁剪作为可选能力保留）                    │
│   ─────────────────────                                             │
│   原理：利用 SDK RunConfig.call_model_input_filter 钩子             │
│   功能：实时统计每次 LLM 调用的 Token 分布与优化效果；**当前实现为    │
│         仅监测、不修改输入**（返回原 model_data），裁剪逻辑作为未启用 │
│         能力保留在代码中，需要时可一行切换                           │
│   效果：提供可观测性；即使前两层配置失效也能及时预警                 │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   优化效果（实测数据，以增量摘要 vs 传统完全替换摘要对比）             │
│   ─────────────────────────────────────────────                     │
│   轮次     传统方式(完全替换)   增量追加方式   节省比例               │
│    2 轮         210 tokens        400 tokens      -                 │
│   10 轮         450 tokens        550 tokens      -                 │
│   50 轮        1650 tokens        550 tokens    66.7% ★             │
│                                                                     │
│   计算方式：                                                         │
│   • 传统方式（50 轮）：基础 150 + 每轮新增 30 ≈ 1650 tokens         │
│   • 增量追加（50 轮）：全局概览 150 + 最近 8 增量 × 50 = 550 tokens │
│     （稳定不再随轮次线性增长）                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**三层优化的实际配置（环境变量）**：

```bash
# 第 1 层：按 Session 类型分别限制历史消息数
SESSION_HISTORY_DEFAULT_LIMIT=20
SESSION_HISTORY_IN_MEMORY_LIMIT=10
SESSION_HISTORY_FEATURE_CONTEXT_LIMIT=30
SESSION_HISTORY_CONVERSATION_CONTEXT_LIMIT=20
SESSION_HISTORY_INFO_COLLECT_LIMIT=20

# 第 2 层：增量摘要
SESSION_HISTORY_ENABLE_SUMMARY_CONTEXT=True
SESSION_HISTORY_SUMMARY_RECENT_LIMIT=10

# 第 3 层：Token 监测
SESSION_HISTORY_ENABLE_TOKEN_MONITORING=True
SESSION_HISTORY_BASELINE_TOKENS=8500
SESSION_HISTORY_TARGET_REDUCTION_PERCENT=59
```

**三层如何注入 SDK Runner：RunConfig 双回调**

第 2 层和第 3 层都是借助 SDK 的 `RunConfig` 钩子接入，互不干扰：

```python
# flow_cortex/multi_agent/core/runtime/context_builder.py
def create_optimized_run_config(session, recent_limit=10, enable_token_filter=True):
    """创建同时支持 Summary 优化和 Token 监测的 RunConfig"""

    # 回调 1：修改上下文结构（注入"全局概览 + 最近增量 + 最近消息"）
    summary_callback = create_summary_callback(session, recent_limit)

    # 回调 2：调用模型前的 Token 监测（当前实现不修改输入，仅记录）
    token_filter = None
    if enable_token_filter and settings.enable_token_monitoring:
        token_filter, _ = create_token_filter()

    return RunConfig(
        session_input_callback=summary_callback,   # 第 2 层：改造输入结构
        call_model_input_filter=token_filter,      # 第 3 层：调用模型前兜底
    )
```

> 两个钩子的关键区别：
>
> - `session_input_callback` 在**读 Session 后、进入 Agent Loop 前**触发——用于主动重塑上下文
> - `call_model_input_filter` 在**每次调用 LLM 前**触发——是最后一道防线

---

## 四、当前方案的局限性

### 4.1 当前方案的局限性

| 问题                           | 描述                                                                                   | 影响                                                      |
| ------------------------------ | -------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **摘要质量依赖 LLM**           | 增量摘要由 LLM 生成，可能丢失关键细节                                                  | 某些场景需要精确信息时可能不足                            |
| **跨 Session 读取开销**        | 每次请求都要加载多个只读 Session                                                       | 高并发场景可能有性能瓶颈                                  |
| **缺乏语义长期记忆**           | 当前摘要是文本压缩，无法语义检索                                                       | 无法回答"之前讨论过 X 吗？"                               |
| **Agent 路由由应用层固定编排** | 放弃了 SDK Handoff 的 LLM 自主路由（见 §1.6/§3.2.4），Agent 调用顺序由外层调度代码写死 | 无法根据上下文动态选择最优 Agent，新增 Agent 需改调度代码 |
| **无情景记忆**                 | Agent 无法从历史执行中学习                                                             | 每次任务从零开始                                          |

### 4.2 核心洞察：记忆系统是 Agent 架构的关键挑战之一

回顾当前方案的五个局限性，会发现一个规律：

```
┌─────────────────────────────────────────────────────────────────────┐
│   局限性                          本质问题                          │
│   ─────────────────────────────────────────────────────────────     │
│   摘要质量依赖 LLM                 → 记忆如何压缩不丢失关键信息？    │
│   缺乏语义长期记忆                 → 记忆如何检索？                  │
│   无情景记忆                       → 记忆存什么？只存事实还是存经验？│
│   跨 Session 读取开销              → 记忆如何高效访问？              │
│   Agent 路由硬编码                 → 记忆如何指导决策？              │
└─────────────────────────────────────────────────────────────────────┘

                              ↓

              所有问题都指向同一个根本约束：
        ┌─────────────────────────────────────────┐
        │   语言模型本身没有状态。                  │
        │   每次调用都从零开始，它不记得任何事情。  │
        └─────────────────────────────────────────┘
```

这意味着：**设计 Agent 系统时，记忆系统的设计是需要优先考虑的核心问题之一。**

记忆系统要回答四个架构问题：**存什么、存在哪、怎么取、怎么管**。不同的答案，决定了完全不同的系统特性。

下一章将分析三个代表性框架（OpenClaw、Claude Code、Hermes Agent）的记忆架构设计，理解它们的核心取舍，为本项目的演进提供参考。

---

## 五、三个明星产品的记忆系统设计

### 5.1 记忆系统要回答的四个架构问题

上一章分析了当前方案的局限性，其中多个问题都指向一个共同的约束：语言模型本身没有状态。这决定了 Agent 框架需要在模型外面搭建记忆系统。

这套系统要回答四个架构问题：

| 问题       | 说明                                     |
| ---------- | ---------------------------------------- |
| **存什么** | 哪些信息值得保留，哪些该丢弃             |
| **存在哪** | 用什么介质，什么格式，什么生命周期       |
| **怎么取** | 需要的时候怎么找到，精确匹配还是语义搜索 |
| **怎么管** | 记忆怎么衰减、更新、压缩，防止积累成噪音 |

OpenClaw、Claude Code、Hermes Agent 对这四个问题给出了三种不同的答案。把它们放在一起看，能看清楚记忆系统设计的核心取舍。

### 5.2 理论框架：记忆的四个层次

研究界把 Agent 记忆分成四种类型，对应不同的存储机制和访问方式：

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Agent 记忆四层架构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   上下文记忆（In-context）                                          │
│   ─────────────────────                                             │
│   当前 Token 窗口里的所有内容                                        │
│   访问成本最低，但容量有限，会话结束即消失                           │
│                                                                     │
│   外部记忆（External）                                               │
│   ─────────────────                                                 │
│   持久化在模型外部的存储——文件、数据库、向量库                       │
│   跨会话存活，但每次访问需要检索步骤                                 │
│                                                                     │
│   情景记忆（Episodic）                                               │
│   ─────────────────                                                 │
│   过去行为的结构化记录                                               │
│   不只是存事实，而是存"做过什么、怎么做的、结果如何"                 │
│   是 Agent 从自身经验学习的基础                                      │
│                                                                     │
│   参数记忆（Parametric）                                             │
│   ───────────────────                                               │
│   模型训练权重里编码的知识                                           │
│   始终存在，不需要检索，但运行时无法更新，也存在幻觉风险             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

真正有趣的架构问题，是**外部记忆**和**情景记忆**怎么设计——这是三个框架差异最大的地方。

### 5.3 OpenClaw：文件系统即记忆

#### 核心设计决策：文件是唯一真理

OpenClaw 的记忆架构建立在一个极简原则上：**没有写进文件的，不存在。**

这不是一句口号，而是一个架构约束。Agent 的所有长期状态，必须持久化到磁盘上的 Markdown 文件里。

```
~/.openclaw/workspace/
├── MEMORY.md              ← 长期记忆（精华）
├── SOUL.md                ← Agent 身份定义
└── memory/
    ├── 2026-04-12.md      ← 当日日志（短期）
    ├── 2026-04-11.md      ← 昨日日志
    └── ...
```

**为什么选择文件而不是数据库？** 这是一个刻意的设计取舍。文件有三个数据库没有的特性：**人类可读、可编辑、可版本控制。**

你可以用任何文本编辑器打开 MEMORY.md，看到 Agent 记住了什么，直接修改错误的记忆，用 Git 追踪变化历史。代价是：文件系统的查询能力远不如数据库。

#### 两层记忆结构：短期与长期的分离

- **短期层** `memory/YYYY-MM-DD.md`：当天工作日志，追加写入，不整理；今天/昨天自动注入，更早靠检索
- **长期层** `MEMORY.md`：从日志沉淀的稳定事实、用户偏好，每次会话都加载

这个两层设计解决了一个根本矛盾：你想要 Agent 记住很多东西，但上下文窗口放不下很多东西。短期层解决"不丢失"，长期层解决"高效访问"。

#### 检索设计：混合搜索

OpenClaw 用**语义搜索 + BM25 关键词搜索**并行：语义搜索处理措辞不同但含义相近的情况，BM25 处理精确词匹配的情况，结果合并取最相关片段。

#### 最危险的环节：Context Compaction

长会话会撑爆 Token 窗口，压缩（把旧历史替换成摘要）是必要操作，但会引入一个隐患：**只存在于对话历史里的约定，会在压缩中消失**。OpenClaw 的解法是 **Memory Flush**——检测到即将 Compaction 时，先触发静默写入把重要信息落到 `memory/YYYY-MM-DD.md`，再执行压缩。**文件里的内容，压缩不会碰**。

### 5.4 Claude Code：上下文工程优先

#### 核心设计决策：Token 预算是稀缺资源

Claude Code 的记忆架构建立在一个明确的工程判断上：

> **上下文窗口的容量不等于可用容量，模型对不同位置的信息注意力分布是不均匀的。**
> 研究表明语言模型对上下文头部和尾部注意力最强，中间最弱——这就是"Lost in the Middle"现象。Claude Code 的记忆架构不是一个"存储系统"，而是一套 **Token 预算分配和信息注入机制**。

#### 系统提示的精确构建：分层注入

系统提示分两层构建：

- **固定注入层**（走 Prefix Cache，只付一次费用）：Agent 身份与行为规范、编码哲学、工具使用规范
- **条件注入层**（按需加载，不浪费 Token）：`CLAUDE.md`（按作用域层级）、Git 状态快照、Skills 索引（仅名称与描述）、Token 预算指令

#### 分层文件体系：用路径编码相关性

```
~/.claude/CLAUDE.md        ← 用户级    ~/project/CLAUDE.md        ← 项目级
                           ~/project/src/CLAUDE.md                ← 目录级
```

不需要写检索算法——当前工作目录在哪，文件系统路径本身就决定加载哪些规则。相关性被编码进目录结构，用 O(1) 的路径查找替代了语义检索。

#### Token 预算的三档预警

**70% → 85% → 90%** 三档阈值逐级提示，最后执行自动压缩。更重要的设计：**Token 使用量会注入 Agent 自身上下文**，让 Agent 在规划任务时能感知剩余预算，主动决策——优先处理哪些文件、在压缩前完成哪些关键步骤。

#### 补充：`memory/` 长期记忆子系统（与 CLAUDE.md 并列的第二条链路）

Claude Code 实际有**两套并行的记忆子系统**，上面讲的是第一套——`CLAUDE.md` 规则/约定层；此外还有第二套 `memory/` 长期记忆层，两者职责互补：

| 维度 | `CLAUDE.md` 层（A） | `memory/` 层（B）                       |
| ---- | ------------------- | --------------------------------------- |
| 性质 | 规则/约定（静态）   | 记忆/偏好（动态）                       |
| 加载 | 路径静态加载，O(1)  | Sonnet 模型做相关性过滤                 |
| 内容 | 项目规范、编码哲学  | 四类：用户偏好 / 反馈 / 项目事实 / 参考 |
| 更新 | 人工维护            | Agent 自动写入，带新鲜度标注            |

**四类分类**：user（用户画像）/ feedback（用户对 Agent 行为的反馈）/ project（项目事实与约定）/ reference（可引用的外部资料）。每条记忆带时间戳与"新鲜度"提示，过期会在召回时被标注。

**与 Hermes Skills 的关键差异**：`memory/` 存的是**偏好与反馈**（"用户不喜欢 emoji"、"这个项目用 pnpm"），不是**经验**（"之前这类任务这样做效果最好"）。也就是说 Claude Code 有偏好型情景记忆、没有经验型情景记忆——这正是它与 Hermes 的分界线。

### 5.5 Hermes Agent：四层分离，情景记忆是核心

#### 核心设计决策：把记忆按访问模式分层

Hermes 的记忆架构是三者中最系统化的。它的核心设计思路是：

> **不同访问模式的记忆，必须在不同的存储介质里，用不同的方式管理。**
> Hermes 把记忆严格分成四层：

**第一层：热记忆（始终注入，永远在场）** — `MEMORY.md`（环境事实，~800 token 上限）+ `USER.md`（用户偏好，~500 token 上限）。上限刻意设得很小——强制做信息的质量控制，避免记忆退化成什么都往里堆的垃圾桶。

**第二层：历史归档（按需检索）** — SQLite + FTS5 全文索引存所有会话历史。不是自动注入，是 Agent 判断任务相关时**主动调用** `session_search` 工具检索。

**第三层：情景记忆（Skills，Hermes 的核心差异）** — 这是 Hermes 和 OpenClaw、Claude Code 最根本的架构差异。Skills 存的不是事实而是**经验**——"做过什么、怎么做的、效果如何"。加载用**渐进式披露**：Level 0 只加载名称和描述，判断相关再加载 Level 1 完整内容。

> **关键特性**：Skill 会在使用中**自我更新**。Agent 用已有 Skill 执行任务时发现更好的做法，会自动修改 Skill 文档。这是真正意义上的情景记忆——不只记住发生了什么，还知道下次怎么做更好。
> **第四层：深度用户建模（可选）** — 通过 Honcho 建立"用户怎么思考、倾向什么决策风格"的持久模型。

### 5.6 三种架构哲学的本质差异

```
┌─────────────────────────────────────────────────────────────────────┐
│                      三种架构哲学对比                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   OpenClaw：文件系统作为真理来源                                     │
│   ─────────────────────────────                                     │
│   设计核心：可见性和可控性                                           │
│   所有记忆都在磁盘上，人类可以直接读写，系统行为透明可预期           │
│   代价：需要用户主动维护记忆质量                                     │
│   适合：重视控制感和透明度的场景                                     │
│                                                                     │
│   Claude Code：上下文工程优先                                        │
│   ─────────────────────────                                         │
│   设计核心：信息的精准调度 + 偏好型记忆                              │
│   CLAUDE.md 做规则注入，memory/ 存用户偏好与反馈                    │
│   代价：没有经验型情景记忆，任务方法论不跨会话积累                   │
│   适合：边界清晰的工程任务                                           │
│                                                                     │
│   Hermes：分层积累，情景记忆是核心                                   │
│   ─────────────────────────────                                     │
│   设计核心：随时间积累能力                                           │
│   四层分离确保不同访问模式的信息不互相干扰                           │
│   代价：系统更复杂，需要时间积累才能看到效果                         │
│   适合：长期运行、重复性高的场景                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 六、框架对比与选型

Agent 的上下文/记忆管理涉及两个不同层面：

```
┌─────────────────────────────────────────────────────────────────────┐
│                      两个层面的框架对比                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  层面一：多 Agent 编排框架                                           │
│  ─────────────────────────                                          │
│  关注：多个 Agent 之间如何共享/隔离上下文，如何协作                  │
│  代表：LangGraph、AutoGen、CrewAI、OpenAI Agents SDK                │
│                                                                     │
│  层面二：单 Agent 记忆系统                                           │
│  ─────────────────────────                                          │
│  关注：单个 Agent 如何管理长期记忆、从经验学习、控制 Token           │
│  代表：OpenClaw、Claude Code、Hermes Agent                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.1 层面一：多 Agent 编排框架对比

| 框架                  | 编排范式      | 上下文共享       | 状态持久化      | 学习曲线 |
| --------------------- | ------------- | ---------------- | --------------- | -------- |
| **LangGraph**         | 图状态机      | State 在图中流转 | Checkpointer    | 中高     |
| **AutoGen**           | 多 Agent 对话 | GroupChat 共享   | save/load_state | 中       |
| **CrewAI**            | 角色扮演      | SharedContext    | 外部数据库      | 低       |
| **OpenAI Agents SDK** | Handoff 原语  | 单 Session 内    | SQLite/Redis    | 低       |

> **本项目的位置**：本项目不是一个新的编排框架，而是**在 OpenAI Agents SDK 之上构建的业务层扩展**——
> 保留 SDK 的 Agent / Runner / Session 原语和 Agent Loop 运行机制，不改动 SDK 源码；仅在 Session 与 Runner 外围补齐
> "多 Agent 上下文隔离 / 跨 Session 声明式共享 / 版本化状态 / Token 三层优化"这四件 SDK 原生未覆盖的事。
> 注：多 Agent 编排未使用 SDK 的 Handoff 原语，而是走应用层调度（见 §1.6、§3.2.4）。
> 因此下文的"选型"讨论的是 SDK 层，本项目是在选定 SDK 之后的**应用层设计**。
> **为什么在四个框架中选择 OpenAI Agents SDK 作为底座**：

| 对比维度       | OpenAI SDK   | LangGraph       | AutoGen           | CrewAI        |
| -------------- | ------------ | --------------- | ----------------- | ------------- |
| 学习曲线       | 低           | 高              | 中                | 低            |
| API 简洁性     | ⭐⭐⭐       | ⭐              | ⭐⭐              | ⭐⭐          |
| Session 内置   | ✅ 多种后端  | 需 Checkpointer | 需自研            | 需自研        |
| 与 OpenAI 集成 | 原生         | 通过 LangChain  | 通过 model_client | 通过 llm 参数 |
| 扩展性         | 中（需自研） | 高              | 高                | 低            |
| 生产就绪       | ✅           | ✅              | ✅                | 中等          |

选择理由：

1. Session 内置 SQLite/Redis/MongoDB 多种后端，减少基础设施开发量；
2. 原语极简（Agent / Runner / Session / Handoff），改造空间大；
3. 核心缺失功能（跨 Session、版本化状态、Token 三层优化）均可在 `RunConfig` 等官方扩展点上自研补齐。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      多 Agent 编排框架详细分析                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LangGraph：图状态机范式                                                 │
│  ─────────────────────                                                  │
│  • 将工作流建模为有向图，节点是处理步骤，边是转移条件                    │
│  • 优点：流程完全可控，支持复杂分支、循环、并行                          │
│  • 缺点：概念多（Node/Edge/State/Checkpoint），学习曲线陡峭              │
│  • 适用：审批流程、决策树、需要精确控制执行顺序的场景                    │
│                                                                         │
│  AutoGen：多 Agent 对话范式                                              │
│  ─────────────────────────                                              │
│  • Agent 之间通过对话协商完成任务，类似"圆桌会议"                        │
│  • 优点：自然的多 Agent 协作，支持 Human-in-the-loop                    │
│  • 缺点：对话轮次难控制，Token 消耗高，可能无限对话                      │
│  • 适用：代码审查、头脑风暴、需要多角度讨论的决策场景                    │
│                                                                         │
│  CrewAI：角色扮演范式                                                    │
│  ───────────────────                                                    │
│  • 模拟真实团队，每个 Agent 有明确角色(Role)和目标(Goal)                 │
│  • 优点：概念直观，上手简单                                              │
│  • 缺点：灵活性受限，状态持久化需自研                                    │
│  • 适用：标准化业务流程、内容生产、有明确分工的场景                      │
│                                                                         │
│  OpenAI Agents SDK：Handoff 原语范式                                     │
│  ─────────────────────────────────                                      │
│  • 通过 Handoff 实现 Agent 间控制权转移，LLM 自主决策                    │
│  • 优点：API 极简，与 OpenAI 生态深度集成，Session 内置                  │
│  • 缺点：复杂编排需自行扩展                                              │
│  • 适用：快速迭代、生产级应用、与 OpenAI 深度集成的场景                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 层面二：单 Agent 记忆系统对比

| 框架            | 外部记忆          | 情景记忆         | Token 控制            |
| --------------- | ----------------- | ---------------- | --------------------- |
| **OpenClaw**    | Markdown + SQLite | Dreaming（实验） | Memory Flush          |
| **Claude Code** | 分层 CLAUDE.md    | ❌ 无            | 三档预警 + Agent 感知 |
| **Hermes**      | SQLite + FTS5     | **Skills** ★     | 渐进式披露            |
| **本实践**      | 数据库            | ❌ 无            | 限制 + 摘要 + 监测    |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      单 Agent 记忆系统详细分析                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  OpenClaw：文件系统作为真理来源                                          │
│  ─────────────────────────────                                          │
│  核心原则："没有写进文件的，不存在"                                       │
│  • 两层分离：短期日志 + 长期 MEMORY.md                                   │
│  • 混合搜索：语义 + BM25 关键词（70%/30% 权重）                          │
│  • Memory Flush：Compaction 前自动写入重要信息                           │
│  • 优势：人类可读、可编辑、透明可控                                       │
│                                                                         │
│  Claude Code：上下文工程优先                                             │
│  ───────────────────────────                                            │
│  核心原则："Token 预算是稀缺资源"                                         │
│  • 分层注入：固定层(Prefix Cache) + 条件层(按需)                         │
│  • 文件层级：用路径编码相关性（用户级/项目级/目录级）                     │
│  • Token 三档预警：70% → 85% → 90%                                       │
│  • 优势：Agent 感知剩余 Token，主动规划任务                               │
│                                                                         │
│  Hermes Agent：四层分离，情景记忆是核心 ★                                 │
│  ─────────────────────────────────────                                  │
│  核心原则："不同访问模式的记忆，必须分层管理"                              │
│  • 第一层：热记忆（始终注入，~1300 字符上限）                             │
│  • 第二层：历史归档（FTS5 全文索引，按需检索）                            │
│  • 第三层：Skills 情景记忆（存经验，Skill 自我更新）                      │
│  • 第四层：深度用户建模（Honcho 辩证推理）                                │
│  • 优势：时间越长 Agent 越熟悉工作方式                                    │
│                                                                         │
│  本项目：三层优化 + 版本化状态 ★                                          │
│  ─────────────────────────────                                          │
│  核心原则："消息与状态分离，跨 Session 可控共享"                          │
│  • 第一层：消息数量限制（默认 20 条）                                     │
│  • 第二层：增量摘要（全局概览 + 最近增量 + 最近消息）                     │
│  • 第三层：Token 监测（RunConfig 钩子）                                   │
│  • 独立机制：版本化 SessionState（状态不随消息清理而丢失）                │
│  • 优势：长对话下 Token 曲线由"线性发散"压为"稳态"（详见 §3.5 数据）│
│  • 待改进：引入情景记忆、Agent 感知 Token 预算                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 选型决策树

```
需要精确控制执行流程？
│
├─ Yes ──▶ 流程复杂度高？
│          ├─ 非常高 ──▶ LangGraph（完全自定义图结构）
│          └─ 一般   ──▶ CrewAI（预设任务流程）
│
└─ No ───▶ 需要 Agent 自主协商？
           ├─ Yes ──▶ AutoGen（多 Agent 对话）
           └─ No ───▶ 需要跨 Session 上下文管理？
                      ├─ Yes ──▶ 本项目方案（SDK + 扩展）✓
                      └─ No ───▶ OpenAI Agents SDK 原生
```

---

**文档版本**: 4.1**创建日期**: 2026-04-19**项目**: 项目

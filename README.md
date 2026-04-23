# LangGraph Agent

一个基于 [LangGraph](https://github.com/langchain-ai/langgraph) 的多 Agent 编排系统，以 supervisor 路由为核心，组合了 **chat / RAG / 工具调用 / 长文创作（小说改编剧本）** 四类 Agent，带有持久化会话历史、向量记忆、LangSmith 追踪与 SSE 流式接口。

---

## 1. 能力速览

- **Supervisor 路由**：规则优先 + LLM 兜底（`plan_routes`），按意图把请求分发给一个或多个 Agent。
- **Chat Agent**：对话式问答、向量记忆检索、历史摘要刷新、follow-up 策略。
- **RAG Agent**：文档语义检索（Chroma），支持稠密 + 关键词混合召回、重排序与改写。
- **Tool Agent**：天气、计算等工具调用的调度与结果回写。
- **Novel Script Agent**：长链 ReAct 图，负责“小说 → 剧本”创作，包含 planner / writer / reviewer 循环与重写预算控制。
- **Merge + Memory**：多 Agent 输出合并，统一写入会话历史与向量记忆。
- **多 Provider LLM**：DeepSeek / GLM / OpenAI，通过 profile 切换（`chat / planner / write / review / embedding`）。
- **可观测性**：结构化 JSON 日志、逐节点 timing、LangSmith trace、debug UI。

---

## 2. 架构概览

```
                 ┌──────────────┐
 user msg  ───▶  │  Supervisor  │  （规则 + LLM 兜底路由）
                 └──────┬───────┘
                        │ routes
        ┌───────────────┼────────────────┬─────────────────┐
        ▼               ▼                ▼                 ▼
   Tool Agent        RAG Agent       Chat Agent     Novel Script Agent
        │               │                │                 │
        └──────┬────────┴────────┬───────┴─────────┬───────┘
               ▼                 ▼                 ▼
                           ┌──────────┐
                           │  Merge   │
                           └────┬─────┘
                                ▼
                        ┌─────────────┐
                        │ Memory Node │  （历史 + 向量记忆 + 摘要）
                        └──────┬──────┘
                               ▼
                              END
```

图定义集中在 `app/graph.py`，状态 schema 在 `app/state.py`。

---

## 3. 目录结构

```
app/
├── api/                 # FastAPI 包：/chat、/chat/stream、/health、/debug-ui
├── agents/
│   ├── supervisor.py    # 路由节点
│   ├── chat/            # 聊天 Agent 子包
│   ├── rag/             # RAG Agent 子包
│   ├── novel_script/    # 长文创作 Agent，内含 react/ 子图
│   ├── tool_agent.py
│   └── merge.py
├── llm/                 # LLM Provider 抽象、缓存、重试、chat/embedding 入口
├── memory/
│   ├── history/         # 会话历史（Protocol 抽象 + SQLite / JSONL 后端）
│   ├── vector_memory.py # Chroma 语义记忆
│   └── write_policy.py
├── retrieval/           # 文档召回、embedding、reranker
├── vector_store/        # Chroma 封装
├── nodes/memory.py      # 后处理：写历史 / 写记忆 / 刷新摘要
├── prompts/             # 提示词模板
├── constants/           # profile 名、路由名、关键词、工具注册
├── utils/               # 日志、tag、错误
├── graph.py             # 主 StateGraph 组装
├── state.py             # AgentState TypedDict
├── chat_service.py      # run_chat_turn 高层入口（CLI / eval 共用）
├── config.py            # RagConfig / MemoryConfig 等配置聚合
├── env.py               # .env 加载 + LangSmith 别名兼容
├── tracing.py           # LangSmith 元数据
└── debug_ui.html

scripts/                 # eval、数据迁移、索引构建等 CLI
data/                    # Chroma 持久化、会话历史 DB/JSONL
outputs/                 # eval 产物（已被 .gitignore 忽略）
```

Stage-1 重构历史：`app/llm.py` / `app/memory/conversation_history.py` / `app/agents/novel_script/graph.py` / `app/api.py` 这 4 个“上帝文件”已全部拆分为独立子包，保留原路径作为门面模块做向后兼容。

---

## 4. 快速开始

### 4.1 环境

- Python 3.10+
- macOS / Linux（Windows 未测试）

### 4.2 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

核心依赖：`langgraph`、`langchain`、`langsmith`、`openai`、`fastapi`、`uvicorn`、`chromadb`、`python-dotenv`。

### 4.3 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填入一个 provider 的 API_KEY
```

最常用的配置项：

| 变量                                                                                 | 含义                                 |
| ------------------------------------------------------------------------------------ | ------------------------------------ |
| `DEEPSEEK_API_KEY` / `GLM_API_KEY` / `OPENAI_API_KEY`                                | LLM provider 密钥（至少一个）        |
| `DEFAULT_CHAT_PROVIDER`                                                              | 普通 chat/创作调用默认 provider      |
| `CREATIVE_PLANNER_PROVIDER` / `CREATIVE_WRITE_PROVIDER` / `CREATIVE_REVIEW_PROVIDER` | 剧本创作链路 3 段的 provider         |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`                                             | embedding profile                    |
| `CHROMA_PERSIST_DIR`                                                                 | 向量库持久化目录，默认 `data/chroma` |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT`                      | 开启 LangSmith 追踪                  |
| `APP_LOG_LEVEL`                                                                      | 结构化日志级别，默认 INFO            |

完整说明见 `.env.example`，分块带中文注释。

### 4.4 启动服务

```bash
uvicorn app.api:app --reload --port 8000
```

- `POST /chat`：同步返回，body：`{ "session_id": "...", "message": "...", "debug": false }`
- `POST /chat/stream`：SSE 流式，事件类型：`start / chunk / done / error / end`
- `GET /health`：健康检查
- `GET /debug-ui`：本地浏览器调试页

### 4.5 CLI 多轮对话

```bash
python -m app.main
```

在终端里 REPL，输入 `exit` 退出。状态会按会话维度保留在进程内。

### 4.6 Docker Compose 本地启动

```bash
cp .env.example .env
# 编辑 .env，至少填入 LLM / embedding 相关 API_KEY

make docker-up
```

服务默认暴露在 `http://127.0.0.1:8000`：

```bash
make docker-health
```

`docker-compose.yml` 会把宿主机目录挂载进容器：

| 宿主机路径 | 容器路径       | 用途                                                         |
| ---------- | -------------- | ------------------------------------------------------------ |
| `./data`   | `/app/data`    | Chroma、conversation history、LangGraph checkpoint 等持久化数据 |
| `./outputs`| `/app/outputs` | eval 产物与本地调试输出                                      |

停止并保留数据：

```bash
make docker-down
```

常用命令：

| 命令                  | 作用                                         |
| --------------------- | -------------------------------------------- |
| `make docker-up`      | 构建镜像并启动服务                           |
| `make docker-down`    | 停止容器，保留 `data/` 和 `outputs/`          |
| `make docker-restart` | 重启服务，验证持久化恢复时很常用             |
| `make docker-logs`    | 跟随查看容器日志                             |
| `make docker-health`  | 检查 `/health`                               |
| `make docker-smoke`   | 自动执行健康检查、请求、重启、总结恢复校验   |
| `make eval-baseline`  | 对运行中的服务执行 baseline eval             |

默认 `.env.example` 里 `LANGSMITH_TRACING=false`，避免 smoke/eval 测试请求污染 LangSmith 项目。需要观测 trace 时，再在 `.env` 里改成 `true`。

如果想一条命令验证 checkpoint 跨容器重启是否生效：

```bash
make docker-smoke
```

它会使用同一个 `session_id` 请求一次，执行 `docker compose restart`，再问总结并检查答案是否包含上一轮问题。

也可以手动验证：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"docker-cp-test","message":"WAI-ARIA技术是什么","debug":true}'

docker compose restart

curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"docker-cp-test","message":"总结所有问题","debug":true}'
```

### 4.7 Docker Compose 单机生产模式

本地模式使用 `./data:/app/data` 这类目录挂载，方便直接查看文件；生产模式使用 Docker 命名 volume，避免把宿主机源码目录和运行数据绑得太死。

```bash
cp .env.example .env
# 编辑 .env，填入真实 API Key，并确认 LANGSMITH_TRACING 是否需要开启

make prod-up
make prod-health
```

生产模式使用 [docker-compose.prod.yml](./docker-compose.prod.yml)：

| Volume 名称          | 容器路径       | 用途                                                         |
| -------------------- | -------------- | ------------------------------------------------------------ |
| `langgraph_data`     | `/app/data`    | Chroma、conversation history、LangGraph checkpoint 等持久化数据 |
| `langgraph_outputs`  | `/app/outputs` | eval 产物与运行输出                                          |

常用生产命令：

| 命令                | 作用                                  |
| ------------------- | ------------------------------------- |
| `make prod-up`      | 构建镜像并后台启动生产 compose         |
| `make prod-down`    | 停止生产容器，保留命名 volume          |
| `make prod-restart` | 重启生产容器                          |
| `make prod-logs`    | 跟随查看生产容器日志                  |
| `make prod-health`  | 检查 `/health`                        |
| `make prod-smoke`   | 在生产 compose 上执行重启持久化 smoke |
| `make prod-backup`  | 备份生产数据 volume 到 `backups/`      |
| `make prod-restore` | 从备份恢复数据 volume，需显式确认      |

生产模式可通过环境变量覆盖镜像名、容器名和端口：

```bash
LANGGRAPH_AGENT_IMAGE=langgraph-agent:v1 \
LANGGRAPH_AGENT_CONTAINER=langgraph-agent \
LANGGRAPH_AGENT_PORT=8000 \
make prod-up
```

命名 volume 不会因为 `make prod-down` 被删除。如果确实要删除数据，需要显式执行 Docker volume 删除命令；这一步有破坏性，执行前请先备份。

生产数据备份：

```bash
# 默认备份 langgraph_data，也就是 /app/data
make prod-backup

# 如果也想备份 outputs volume
VOLUME_NAME=langgraph_outputs make prod-backup
```

备份文件会写到 `backups/`，文件名类似：

```text
backups/langgraph_data-20260423-120000.tar.gz
```

恢复会清空目标 volume 后再解压备份，必须显式确认：

```bash
make prod-down
BACKUP_PATH=backups/langgraph_data-20260423-120000.tar.gz \
CONFIRM_RESTORE=yes \
make prod-restore
make prod-up
```

恢复 `langgraph_outputs` 时同理传入 `VOLUME_NAME=langgraph_outputs`。恢复是破坏性操作，建议先对当前 volume 再做一次备份。

---

## 5. 评测与观测

### 5.1 离线评测

```bash
# 本地内嵌模式（使用 FastAPI TestClient）
python scripts/eval_chat.py

# 远程模式
EVAL_BASE_URL=http://localhost:8000 python scripts/eval_chat.py

# 只跑指定 case
EVAL_CASE_IDS=case_01,case_05 python scripts/eval_chat.py
```

测试用例在 `scripts/eval_cases.json`，产物落到 `outputs/eval_runs/`。

相关脚本：

- `scripts/run_eval_profile.py`：跨 profile A/B 对比
- `scripts/compare_eval_runs.py`：两次 run 的 diff
- `scripts/build_doc_embeddings.py` + `scripts/build_doc_chroma_index.py`：文档向量化 & 建索引
- `scripts/clear_conversation_history.py`：清理历史
- `scripts/migrate_*.py` / `scripts/inspect_*.py`：数据迁移与巡检

### 5.2 LangSmith 追踪

将 `LANGSMITH_TRACING=true` 并配置 `LANGSMITH_API_KEY` 后，每次调用会自动带上 request_id、session_id、route 等元数据，方便在 LangSmith 上按会话/路由聚合查看。默认建议保持关闭，需要排查链路或观察模型调用时再开启。

### 5.3 本地日志

应用日志为 **单行 JSON**，直接适配容器日志采集。示例字段：`ts`、`level`、`event`、`session_id`、`request_id`、`node`、`duration_ms`。

---

## 6. 数据存储

- **会话历史**：`app/memory/history/`，支持两种后端，通过配置切换：
  - `sqlite_backend.py`：WAL 模式 + busy_timeout，适合并发读写
  - `jsonl_backend.py`：纯文件、易检查、适合小规模 / 调试
- **向量记忆**：Chroma，持久化目录由 `CHROMA_PERSIST_DIR` 控制
- **文档索引**：同一 Chroma 实例下的独立 collection（`CHROMA_DOC_COLLECTION`）

---

## 7. 开发约定

- **并发 / 锁序**：Session 状态访问遵循 `store_guard(短) → session_lock(长) → store_guard(短)`。逻辑封装在 `app/api/session_store.py` 与 `app/api/chat_runner.py._invoke_with_session_lock`，**不要**在 guard 中直接执行 graph.invoke。
- **LLM 调用**：统一走 `app/llm/` 的 `chat()` / `chat_with_tools()` / `embed_text()`，不要绕过去直连 `openai` SDK；所有调用都由重试 + 超时 + request 级缓存包裹。
- **新增 Agent**：
  1. 在 `app/agents/` 下新建子包，暴露 `xxx_agent_node(state) -> state_delta`
  2. 在 `app/constants/routes.py` 新增 `ROUTE_*` 常量
  3. 在 `app/graph.py` 注册节点与条件边
  4. 如有路由特征，在 `app/agents/supervisor.py` 添加规则或更新 planner 提示词
- **新增小说创作工具**：在 `app/agents/novel_script/react/tool_dispatch.py` 的 `TOOLS` + `TOOL_REDUCERS` 两张表中登记即可，`nodes.py` 不需要改。
- **提示词**：统一放在 `app/prompts/`，避免散落在业务代码里。

---

## 8. License

仅用于个人学习与技术验证，未设定开源协议。若需二次使用请先联系作者。

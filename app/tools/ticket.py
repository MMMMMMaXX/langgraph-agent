"""`ticket_create` 工具：一个真正有副作用的 mock 下游。

现实里这个位置应该是对接工单系统（Jira / ServiceNow）的 HTTP 客户端；
Phase 1 先用本地 SQLite 表 `mock_tickets` 模拟"已经在下游落地"的语义，
让 tool_safety 的两层防护（execution_record + UNIQUE 约束）能被端到端
验证：
- execution_record：idempotency_key 在 `tool_executions` PRIMARY KEY 拦截同一
  请求重入；
- UNIQUE：即便 execution_record 层漏判（换机器、缓存失效），下游表本身也
  通过 `idempotency_key UNIQUE` 兜底，避免重复工单。

接口设计：pipeline 层会在调用时额外注入 `idempotency_key / tenant_id /
user_id`，真实工具接这些隐藏字段做写入；LLM 暴露的 schema 只有业务
字段（`title` / `description`）。
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import OPERATIONS_CONFIG

CREATE_MOCK_TICKETS_SQL = """
CREATE TABLE IF NOT EXISTS mock_tickets (
    ticket_id        TEXT PRIMARY KEY,
    idempotency_key  TEXT NOT NULL UNIQUE,
    tenant_id        TEXT NOT NULL,
    user_id          TEXT NOT NULL,
    title            TEXT NOT NULL,
    description      TEXT,
    created_at       INTEGER NOT NULL
)
"""

CREATE_INDEX_TENANT_USER_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_mock_tickets_tenant_user "
    "ON mock_tickets(tenant_id, user_id)"
)


def _default_path() -> Path:
    return Path(OPERATIONS_CONFIG.path)


def _connect(path: Path | None = None) -> sqlite3.Connection:
    """Short connection + WAL；schema 随连接建立自动就位，无独立 migration。"""

    resolved = _default_path() if path is None else Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute(CREATE_MOCK_TICKETS_SQL)
    conn.execute(CREATE_INDEX_TENANT_USER_SQL)
    return conn


def create_ticket(
    *,
    title: str,
    description: str = "",
    idempotency_key: str,
    tenant_id: str,
    user_id: str,
    path: Path | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    """创建工单；如果 idempotency_key 已经存在，返回既有工单 + `created=False`。

    二次请求命中 UNIQUE 不当异常，因为这是"幂等重放成功"的正常语义；
    调用方（pipeline）已经用 execution_record 做过第一层拦截，本函数只是
    下游对同一意图的再一次拒绝写入。
    """

    if not title:
        raise ValueError("title must not be empty")
    if not idempotency_key:
        raise ValueError("idempotency_key must not be empty")

    timestamp = now if now is not None else int(time.time())
    ticket_id = f"TK-{uuid.uuid4().hex[:12]}"

    with _connect(path) as conn:
        try:
            conn.execute(
                """
                INSERT INTO mock_tickets (
                    ticket_id, idempotency_key, tenant_id, user_id,
                    title, description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    idempotency_key,
                    tenant_id,
                    user_id,
                    title,
                    description,
                    timestamp,
                ),
            )
            return {
                "ticket_id": ticket_id,
                "created": True,
                "idempotency_key": idempotency_key,
            }
        except sqlite3.IntegrityError as exc:
            if "idempotency_key" not in str(exc) and "UNIQUE" not in str(exc).upper():
                raise
            # 命中幂等键：把既有 ticket 返回出去，语义是"此前已创建成功"。
            existing = conn.execute(
                "SELECT ticket_id FROM mock_tickets WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is None:
                # 理论不可能：刚报 UNIQUE 冲突却读不到记录，视为数据库异常冒泡。
                raise RuntimeError(
                    f"mock_tickets UNIQUE conflict without row: {idempotency_key}"
                ) from exc
            return {
                "ticket_id": existing["ticket_id"],
                "created": False,
                "idempotency_key": idempotency_key,
            }


def ticket_create_tool(
    *,
    title: str,
    description: str = "",
    idempotency_key: str = "",
    tenant_id: str = "",
    user_id: str = "",
    **_: Any,
) -> dict[str, Any]:
    """LLM function-calling 入口。

    Pipeline 在调用时会通过 kwargs 注入 `idempotency_key / tenant_id / user_id`；
    如果哪个 caller 漏注入（例如写测试时绕过 pipeline 直接调），这里立刻报错，
    避免静默把工单写到匿名维度。
    """

    if not idempotency_key or not tenant_id or not user_id:
        raise ValueError(
            "ticket_create must be invoked through side_effect pipeline: "
            "missing idempotency_key / tenant_id / user_id"
        )
    return create_ticket(
        title=title,
        description=description,
        idempotency_key=idempotency_key,
        tenant_id=tenant_id,
        user_id=user_id,
    )


__all__ = [
    "CREATE_INDEX_TENANT_USER_SQL",
    "CREATE_MOCK_TICKETS_SQL",
    "create_ticket",
    "ticket_create_tool",
]

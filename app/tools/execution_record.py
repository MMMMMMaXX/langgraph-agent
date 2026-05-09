"""`tool_executions` 表的 CRUD。

职责边界：
- 本模块**只**关心 SQLite 读写 + schema 初始化；不知道 idempotency_key
  怎么算、不知道工具 metadata。
- 上层 `app/tools/idempotency.py` 负责业务语义（抢占式 INSERT、轮询、
  状态机迁移）。这样后续把 SQLite 换成 Postgres，只需要替换本模块。

设计要点：
- 连接模式参考 `app/memory/history/sqlite_backend.py`：短连接 + WAL +
  `CREATE TABLE IF NOT EXISTS`，运行前不需要独立 migration 步骤。
- `idempotency_key` 做 PRIMARY KEY：winner-takes-all 写入依赖主键冲突，
  本模块向上抛 `ExecutionRecordAlreadyExists` 让上层判断。
- `args_json` / `result_json` 以字符串形式存，不解析；所有序列化细节由
  调用方控制，保证"存进去什么，取出来就是什么"。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import OPERATIONS_CONFIG
from app.constants.tool_safety import (
    TOOL_STATUS_PENDING,
    VALID_TOOL_STATUSES,
)

# ---------- schema ----------

CREATE_TOOL_EXECUTIONS_SQL = """
CREATE TABLE IF NOT EXISTS tool_executions (
    idempotency_key TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    request_id      TEXT NOT NULL,
    tool_name       TEXT NOT NULL,
    args_json       TEXT NOT NULL,
    status          TEXT NOT NULL,
    result_json     TEXT,
    error           TEXT,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
)
"""

CREATE_INDEX_SESSION_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_tool_exec_session " "ON tool_executions(session_id)"
)
CREATE_INDEX_USER_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_tool_exec_user "
    "ON tool_executions(tenant_id, user_id)"
)


# ---------- dataclass ----------


@dataclass(frozen=True)
class ExecutionRecord:
    """`tool_executions` 一行对应的只读视图。"""

    idempotency_key: str
    tenant_id: str
    user_id: str
    session_id: str
    request_id: str
    tool_name: str
    args_json: str
    status: str
    result_json: str | None
    error: str | None
    created_at: int
    updated_at: int

    @property
    def args(self) -> Any:
        """`args_json` 的反序列化视图。仅在上层确实需要时才调用。"""

        return json.loads(self.args_json)

    @property
    def result(self) -> Any:
        if self.result_json is None:
            return None
        return json.loads(self.result_json)


class ExecutionRecordAlreadyExists(Exception):
    """`insert_pending` 因 idempotency_key 主键冲突失败时抛出。

    上层应捕获本异常，再 `get_by_key` 读取已有 record 决定语义。
    """

    def __init__(self, idempotency_key: str) -> None:
        super().__init__(f"execution record already exists: {idempotency_key}")
        self.idempotency_key = idempotency_key


# ---------- helpers ----------


def _default_path() -> Path:
    return Path(OPERATIONS_CONFIG.path)


def _connect(path: Path | None = None) -> sqlite3.Connection:
    """打开连接并确保 schema 就位。

    这里不缓存连接，让每次调用都是短生命周期事务，避免跨线程共享 SQLite
    连接的 `check_same_thread` 问题——Phase 1 的调用量完全可以承受。
    """

    resolved = _default_path() if path is None else Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute(CREATE_TOOL_EXECUTIONS_SQL)
    conn.execute(CREATE_INDEX_SESSION_SQL)
    conn.execute(CREATE_INDEX_USER_SQL)
    return conn


def _row_to_record(row: sqlite3.Row) -> ExecutionRecord:
    return ExecutionRecord(
        idempotency_key=row["idempotency_key"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        request_id=row["request_id"],
        tool_name=row["tool_name"],
        args_json=row["args_json"],
        status=row["status"],
        result_json=row["result_json"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _now() -> int:
    return int(time.time())


# ---------- public API ----------


def insert_pending(
    *,
    idempotency_key: str,
    tenant_id: str,
    user_id: str,
    session_id: str,
    request_id: str,
    tool_name: str,
    args_json: str,
    path: Path | None = None,
    now: int | None = None,
) -> ExecutionRecord:
    """以 `status='pending'` 插入一行。

    返回新插入的 `ExecutionRecord`。主键冲突抛 `ExecutionRecordAlreadyExists`，
    表示当前 idempotency_key 已经被抢占，调用方应转向 `get_by_key` 读已有记录。
    """

    if not idempotency_key:
        raise ValueError("idempotency_key must not be empty")

    timestamp = now if now is not None else _now()
    with _connect(path) as conn:
        try:
            conn.execute(
                """
                INSERT INTO tool_executions (
                    idempotency_key, tenant_id, user_id, session_id,
                    request_id, tool_name, args_json, status,
                    result_json, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    idempotency_key,
                    tenant_id,
                    user_id,
                    session_id,
                    request_id,
                    tool_name,
                    args_json,
                    TOOL_STATUS_PENDING,
                    timestamp,
                    timestamp,
                ),
            )
        except sqlite3.IntegrityError as exc:
            # IntegrityError 覆盖主键冲突，也覆盖 NOT NULL / UNIQUE 违反；
            # 只有主键冲突属于"正常竞争"语义，其他一律当作编程错误冒泡。
            if "idempotency_key" in str(exc) or "PRIMARY KEY" in str(exc).upper():
                raise ExecutionRecordAlreadyExists(idempotency_key) from exc
            raise

    return ExecutionRecord(
        idempotency_key=idempotency_key,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        tool_name=tool_name,
        args_json=args_json,
        status=TOOL_STATUS_PENDING,
        result_json=None,
        error=None,
        created_at=timestamp,
        updated_at=timestamp,
    )


def get_by_key(
    idempotency_key: str,
    path: Path | None = None,
) -> ExecutionRecord | None:
    """按 idempotency_key 取单行。未命中返回 None。"""

    with _connect(path) as conn:
        row = conn.execute(
            "SELECT * FROM tool_executions WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
    return _row_to_record(row) if row is not None else None


def update_status(
    *,
    idempotency_key: str,
    status: str,
    result_json: str | None = None,
    error: str | None = None,
    path: Path | None = None,
    now: int | None = None,
) -> ExecutionRecord:
    """迁移状态机：`pending` → 终态。

    - 校验 `status` 属于合法集合，且**不**是 pending（pending 只能由 insert
      产生，避免"UPDATE 回 pending"打穿状态机）。
    - 未命中 idempotency_key 时抛 KeyError，交由上层决定是否是 bug。
    """

    if status not in VALID_TOOL_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    if status == TOOL_STATUS_PENDING:
        raise ValueError("update_status must not set status back to pending")

    timestamp = now if now is not None else _now()
    with _connect(path) as conn:
        cursor = conn.execute(
            """
            UPDATE tool_executions
               SET status = ?,
                   result_json = ?,
                   error = ?,
                   updated_at = ?
             WHERE idempotency_key = ?
            """,
            (status, result_json, error, timestamp, idempotency_key),
        )
        if cursor.rowcount == 0:
            raise KeyError(idempotency_key)

    record = get_by_key(idempotency_key, path=path)
    assert record is not None  # 刚 UPDATE 成功，理论上一定能读到
    return record


__all__ = [
    "CREATE_INDEX_SESSION_SQL",
    "CREATE_INDEX_USER_SQL",
    "CREATE_TOOL_EXECUTIONS_SQL",
    "ExecutionRecord",
    "ExecutionRecordAlreadyExists",
    "get_by_key",
    "insert_pending",
    "update_status",
]

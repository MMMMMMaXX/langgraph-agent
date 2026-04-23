"""SQLite 后端实现。

单机开发默认后端，提供：
- WAL + 短连接，让并发读写尽量不互相阻塞
- 两条索引支撑按 session 查询和窗口内去重
- 每次请求复用一个事务，异常时自动回滚
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.utils.logger import now_timestamp_s

from .events import make_dedupe_stub
from .schema import (
    CREATE_DEDUPE_INDEX_SQL,
    CREATE_HISTORY_TABLE_SQL,
    CREATE_SESSION_TIME_INDEX_SQL,
    DEFAULT_HISTORY_SOURCE,
    HISTORY_EVENT_VERSION,
    SQLITE_BUSY_TIMEOUT_MS,
    SQLITE_JOURNAL_MODE_WAL,
    SQLITE_SYNCHRONOUS_NORMAL,
    json_dumps,
    json_loads_list,
    normalize_history_question,
    normalize_routes,
)


class SQLiteBackend:
    """基于 sqlite3 标准库的 HistoryBackend 实现。"""

    def __init__(self, dedupe_window_seconds: int) -> None:
        self.dedupe_window_seconds = dedupe_window_seconds

    # ---------- connection ----------

    def _connect(self, path: Path) -> sqlite3.Connection:
        """打开 SQLite 连接并确保 schema 已初始化。

        SQLite 是单文件数据库，不需要单独启动服务。这里每次打开连接时都执行
        CREATE TABLE/INDEX IF NOT EXISTS，保证首次运行、eval 隔离文件、迁移文件
        都能自动完成初始化。

        row_factory=sqlite3.Row 让查询结果可以通过 row["字段名"] 读取，
        比按下标取值更不容易在 schema 变动时出错。
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        # WAL 让读写并发更友好：读请求不会阻塞正在追加的写事务。
        # busy_timeout 则让短时间写锁竞争先等待，而不是立刻抛 database is locked。
        # history 属于可恢复的会话流水，NORMAL 能在 WAL 下减少频繁 fsync 开销。
        conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        conn.execute(f"PRAGMA journal_mode = {SQLITE_JOURNAL_MODE_WAL}")
        conn.execute(f"PRAGMA synchronous = {SQLITE_SYNCHRONOUS_NORMAL}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(CREATE_HISTORY_TABLE_SQL)
        conn.execute(CREATE_SESSION_TIME_INDEX_SQL)
        conn.execute(CREATE_DEDUPE_INDEX_SQL)
        return conn

    # ---------- row mapping ----------

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
        """把 SQLite row 转回主链路使用的 event dict。

        SQLite 里没有直接存 Python list，这里把 routes_json / tags_json
        反序列化回列表，避免上层关心数据库存储细节。
        """

        return {
            "id": row["id"],
            "version": row["version"],
            "session_id": row["session_id"],
            "timestamp": row["timestamp"],
            "source": row["source"],
            "routes": json_loads_list(row["routes_json"]),
            "user_message": row["user_message"],
            "rewritten_query": row["rewritten_query"],
            "normalized_question": row["normalized_question"],
            "answer_preview": row["answer_preview"],
            "tags": json_loads_list(row["tags_json"]),
            "stored_to_vector": bool(row["stored_to_vector"]),
            "skipped_vector_store": bool(row["skipped_vector_store"]),
            "vector_store_skip_reason": row["vector_store_skip_reason"],
            "skipped_duplicate": bool(row["skipped_duplicate"]),
            "created_at": row["created_at"],
        }

    # ---------- dedupe & insert ----------

    def _is_duplicate(
        self,
        conn: sqlite3.Connection,
        event: dict[str, Any],
    ) -> bool:
        """用索引判断是否重复。

        查询条件对应 idx_history_dedupe：
        session_id + normalized_question + routes_json + timestamp
        """

        if self.dedupe_window_seconds <= 0:
            return False

        routes_json = json_dumps(normalize_routes(event.get("routes", [])))
        threshold = float(event["timestamp"]) - self.dedupe_window_seconds
        row = conn.execute(
            """
            SELECT id
            FROM conversation_history
            WHERE session_id = ?
              AND normalized_question = ?
              AND routes_json = ?
              AND timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (
                event["session_id"],
                event["normalized_question"],
                routes_json,
                threshold,
            ),
        ).fetchone()
        return row is not None

    @staticmethod
    def _insert(conn: sqlite3.Connection, event: dict[str, Any]) -> None:
        """插入一条 event。

        这里使用参数化 SQL 的 ? 占位符，而不是字符串拼接：
        - 避免用户输入里的引号/换行破坏 SQL
        - 避免 SQL 注入

        routes/tags 作为 JSON 字符串存储，是为了在保持 schema 简单的同时
        保留列表结构；读取时再 json.loads 还原。
        """

        conn.execute(
            """
            INSERT INTO conversation_history (
              version,
              session_id,
              timestamp,
              source,
              routes_json,
              user_message,
              rewritten_query,
              normalized_question,
              answer_preview,
              tags_json,
              stored_to_vector,
              skipped_vector_store,
              vector_store_skip_reason,
              skipped_duplicate
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(event.get("version") or HISTORY_EVENT_VERSION),
                event.get("session_id", ""),
                float(event.get("timestamp") or now_timestamp_s()),
                event.get("source", DEFAULT_HISTORY_SOURCE),
                json_dumps(normalize_routes(event.get("routes", []))),
                event.get("user_message", ""),
                event.get("rewritten_query", ""),
                event.get("normalized_question")
                or normalize_history_question(
                    event.get("user_message", ""),
                    event.get("rewritten_query", ""),
                ),
                event.get("answer_preview", ""),
                json_dumps(event.get("tags", [])),
                1 if event.get("stored_to_vector") else 0,
                1 if event.get("skipped_vector_store") else 0,
                event.get("vector_store_skip_reason", ""),
                1 if event.get("skipped_duplicate") else 0,
            ),
        )

    # ---------- HistoryBackend interface ----------

    def append(
        self,
        event: dict[str, Any],
        path: Path,
        dedupe: bool,
    ) -> dict[str, Any]:
        """写入前先用索引做短窗口去重；不是重复则 INSERT。

        每次请求使用短连接，代码简单且适合当前单机开发场景。
        """

        with self._connect(path) as conn:
            if dedupe and self._is_duplicate(conn, event):
                return make_dedupe_stub(event)
            self._insert(conn, event)
        return event

    def read_session(
        self,
        session_id: str,
        limit: int,
        path: Path,
    ) -> list[dict[str, Any]]:
        """按 idx_history_session_time 倒序取最近 limit 条，再反转为正序返回。"""

        capped = max(limit, 0)
        if capped == 0:
            return []
        with self._connect(path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM conversation_history
                WHERE session_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (session_id, capped),
            ).fetchall()
        return list(reversed([self._row_to_event(row) for row in rows]))

    def read_all(self, path: Path) -> list[dict[str, Any]]:
        """全表读取，仅用于治理脚本。"""

        with self._connect(path) as conn:
            rows = conn.execute(
                "SELECT * FROM conversation_history ORDER BY timestamp ASC, id ASC"
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def write_all(self, events: list[dict[str, Any]], path: Path) -> None:
        """清空后重新写入。

        with sqlite3 connection 会形成事务：正常退出自动 commit，
        异常时自动 rollback，避免只写入一半。
        """

        with self._connect(path) as conn:
            conn.execute("DELETE FROM conversation_history")
            for event in events:
                self._insert(conn, event)

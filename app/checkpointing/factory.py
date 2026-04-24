from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import CHECKPOINT_CONFIG

_CHECKPOINTER = None

SQLITE_BUSY_TIMEOUT_MS = 5000

OFFICIAL_CHECKPOINT_COLUMNS = {
    "thread_id",
    "checkpoint_ns",
    "checkpoint_id",
    "parent_checkpoint_id",
    "type",
    "checkpoint",
    "metadata",
}

LEGACY_CHECKPOINT_TABLES = (
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _checkpoint_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(checkpoints)")}


def _backup_legacy_checkpoint_tables(conn: sqlite3.Connection) -> None:
    """发现旧版自定义 checkpoint 表结构时，先重命名备份再交给官方建表。"""

    if not _table_exists(conn, "checkpoints"):
        return
    if OFFICIAL_CHECKPOINT_COLUMNS.issubset(_checkpoint_columns(conn)):
        return

    suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    for table_name in LEGACY_CHECKPOINT_TABLES:
        if _table_exists(conn, table_name):
            conn.execute(
                f"ALTER TABLE {table_name} RENAME TO {table_name}_legacy_{suffix}"
            )
    conn.commit()


def _build_sqlite_saver(path: str) -> SqliteSaver:
    """创建官方 SqliteSaver，并补齐本项目需要的连接级 SQLite 参数。"""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    _backup_legacy_checkpoint_tables(conn)
    return SqliteSaver(conn)


def build_checkpointer():
    """构建 LangGraph checkpointer。

    默认使用 SQLite，让本地开发、Docker 单实例部署都能跨进程重启恢复 session。
    如需临时关闭持久化，可设置 LANGGRAPH_CHECKPOINT_ENABLED=false，
    此时回退到 InMemorySaver，便于排查 SQLite 文件问题。
    """

    if not CHECKPOINT_CONFIG.enabled:
        return InMemorySaver()
    return _build_sqlite_saver(CHECKPOINT_CONFIG.path)


def get_checkpointer():
    """返回进程内共享 checkpointer 实例。"""

    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        _CHECKPOINTER = build_checkpointer()
    return _CHECKPOINTER


def clear_checkpoints() -> None:
    """清空当前 checkpointer，主要用于 eval/test 隔离。"""

    checkpointer = get_checkpointer()
    conn = getattr(checkpointer, "conn", None)
    if conn is not None:
        checkpointer.setup()
        with checkpointer.cursor() as cursor:
            cursor.execute("DELETE FROM writes")
            cursor.execute("DELETE FROM checkpoints")
        return

    # InMemorySaver 没有统一 clear_all API，这里按其公开属性做测试场景清理。
    for attr in ("storage", "writes", "blobs"):
        value = getattr(checkpointer, attr, None)
        if value is not None:
            value.clear()

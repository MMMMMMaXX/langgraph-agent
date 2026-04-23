"""History 存储相关的常量、DDL 以及通用的序列化/规范化工具。

这些工具不关心底层是 SQLite 还是 JSONL，业务层（events.py）和
各 backend 实现都会复用它们，保证同一份 event 字典在不同后端下
拥有一致的规范化结果（routes 排序、tags 去空白、question 标准化）。
"""

from __future__ import annotations

import json
from typing import Any

HISTORY_BACKEND_SQLITE = "sqlite"
HISTORY_BACKEND_JSONL = "jsonl"
HISTORY_EVENT_VERSION = 1
DEFAULT_HISTORY_SOURCE = "chat_round"
ANSWER_PREVIEW_CHARS = 500
HISTORY_PREVIEW_CHARS = 100
DEFAULT_DEDUPE_ENABLED = True

SQLITE_SUFFIXES = (".sqlite", ".sqlite3", ".db")
JSONL_SUFFIX = ".jsonl"

SQLITE_BUSY_TIMEOUT_MS = 5000
SQLITE_JOURNAL_MODE_WAL = "wal"
SQLITE_SYNCHRONOUS_NORMAL = "NORMAL"

CREATE_HISTORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  version INTEGER NOT NULL,
  session_id TEXT NOT NULL,
  timestamp REAL NOT NULL,
  source TEXT NOT NULL,
  routes_json TEXT NOT NULL,
  user_message TEXT NOT NULL,
  rewritten_query TEXT NOT NULL,
  normalized_question TEXT NOT NULL,
  answer_preview TEXT NOT NULL,
  tags_json TEXT NOT NULL,
  stored_to_vector INTEGER NOT NULL DEFAULT 0,
  skipped_vector_store INTEGER NOT NULL DEFAULT 0,
  vector_store_skip_reason TEXT NOT NULL DEFAULT '',
  skipped_duplicate INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_SESSION_TIME_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_history_session_time
ON conversation_history(session_id, timestamp)
"""

CREATE_DEDUPE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_history_dedupe
ON conversation_history(session_id, normalized_question, routes_json, timestamp)
"""


def normalize_tags(tags: list[str] | tuple[str, ...] | None) -> list[str]:
    """去掉 tags 里的空白值，保留输入顺序。"""

    if not tags:
        return []
    return [str(tag).strip() for tag in tags if str(tag).strip()]


def normalize_routes(routes: list[str] | tuple[str, ...] | None) -> list[str]:
    """规范化 routes：去空白 + 字典序排序。

    去重对比要求“routes 集合相同”，排序后保证 ["rag","novel"] 和
    ["novel","rag"] 产生相同的 JSON 字符串，索引命中一致。
    """

    if not routes:
        return []
    return sorted(str(route).strip() for route in routes if str(route).strip())


def normalize_history_question(user_message: str, rewritten_query: str = "") -> str:
    """生成用于去重和总结展示的标准化问题。

    rewrite 后的问题通常比原始追问更完整；去掉结尾问号可以让
    “WAI-ARIA技术是什么”和“WAI-ARIA技术是什么？”视为同一问题。
    """

    question = rewritten_query or user_message
    return str(question).strip().rstrip("？?")


def json_dumps(value: Any) -> str:
    """紧凑 JSON 序列化，routes_json/tags_json 使用同一规则才能走索引。"""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_loads_list(value: str) -> list:
    """安全反序列化为 list，失败或非 list 时返回空列表。"""

    try:
        loaded = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []

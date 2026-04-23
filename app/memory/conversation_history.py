import json
import sqlite3
from pathlib import Path
from typing import Any

from app.config import CONVERSATION_HISTORY_CONFIG
from app.utils.logger import now_timestamp_s, preview

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


def resolve_history_backend(
    backend: str | None = None,
    history_path: str | None = None,
) -> str:
    """确定本次 history 操作使用哪个后端。

    优先级：
    1. 调用方显式传入 backend
    2. 根据请求级 history_path 后缀推断，比如 .sqlite3 / .jsonl
    3. 使用全局配置 CONVERSATION_HISTORY_BACKEND

    这样 eval 可以通过请求级 path 临时切到隔离 SQLite 文件，
    主服务也可以继续使用默认配置。
    """

    if backend:
        return backend.lower()

    if history_path:
        suffix = Path(history_path).suffix.lower()
        if suffix == JSONL_SUFFIX:
            return HISTORY_BACKEND_JSONL
        if suffix in SQLITE_SUFFIXES:
            return HISTORY_BACKEND_SQLITE

    configured_backend = CONVERSATION_HISTORY_CONFIG.backend.lower()
    if configured_backend in {HISTORY_BACKEND_SQLITE, HISTORY_BACKEND_JSONL}:
        return configured_backend
    return HISTORY_BACKEND_SQLITE


def resolve_history_path(
    backend: str | None = None,
    history_path: str | None = None,
) -> Path:
    """确定本次 history 操作读写哪个文件。

    history_path 是请求级覆盖，主要用于 eval 隔离；
    没有覆盖时再按 backend 选择默认 sqlite_path / jsonl_path。
    """

    if history_path:
        return Path(history_path)

    resolved_backend = resolve_history_backend(backend)
    if resolved_backend == HISTORY_BACKEND_JSONL:
        return Path(CONVERSATION_HISTORY_CONFIG.jsonl_path)
    return Path(CONVERSATION_HISTORY_CONFIG.sqlite_path)


def _normalize_tags(tags: list[str] | tuple[str, ...] | None) -> list[str]:
    if not tags:
        return []
    return [str(tag).strip() for tag in tags if str(tag).strip()]


def _normalize_routes(routes: list[str] | tuple[str, ...] | None) -> list[str]:
    if not routes:
        return []
    return sorted(str(route).strip() for route in routes if str(route).strip())


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads_list(value: str) -> list:
    try:
        loaded = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


def normalize_history_question(user_message: str, rewritten_query: str = "") -> str:
    """生成用于去重和总结展示的标准化问题。

    rewrite 后的问题通常比原始追问更完整；去掉结尾问号可以让
    “WAI-ARIA技术是什么”和“WAI-ARIA技术是什么？”视为同一问题。
    """

    question = rewritten_query or user_message
    return str(question).strip().rstrip("？?")


def build_history_event(
    *,
    session_id: str,
    user_message: str,
    answer: str,
    rewritten_query: str = "",
    routes: list[str] | None = None,
    tags: list[str] | None = None,
    stored_to_vector: bool = False,
    skipped_vector_store: bool = False,
    vector_store_skip_reason: str = "",
    source: str = DEFAULT_HISTORY_SOURCE,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """把主链路状态整理成统一 history event。

    不论底层写 SQLite 还是 JSONL，业务层都先构造同一份 dict。
    这样后端切换不会影响 memory_node / chat_agent 的调用方式。
    """

    current_time = now_timestamp_s() if timestamp is None else timestamp
    return {
        "version": HISTORY_EVENT_VERSION,
        "session_id": session_id,
        "timestamp": current_time,
        "source": source,
        "routes": routes or [],
        "user_message": user_message,
        "rewritten_query": rewritten_query,
        "normalized_question": normalize_history_question(
            user_message, rewritten_query
        ),
        "answer_preview": preview(answer, ANSWER_PREVIEW_CHARS),
        "tags": _normalize_tags(tags),
        "stored_to_vector": stored_to_vector,
        "skipped_vector_store": skipped_vector_store,
        "vector_store_skip_reason": vector_store_skip_reason,
        "skipped_duplicate": False,
    }


def _connect_sqlite(path: Path) -> sqlite3.Connection:
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
        "routes": _json_loads_list(row["routes_json"]),
        "user_message": row["user_message"],
        "rewritten_query": row["rewritten_query"],
        "normalized_question": row["normalized_question"],
        "answer_preview": row["answer_preview"],
        "tags": _json_loads_list(row["tags_json"]),
        "stored_to_vector": bool(row["stored_to_vector"]),
        "skipped_vector_store": bool(row["skipped_vector_store"]),
        "vector_store_skip_reason": row["vector_store_skip_reason"],
        "skipped_duplicate": bool(row["skipped_duplicate"]),
        "created_at": row["created_at"],
    }


def _read_jsonl_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                # JSONL 是追加写，单行损坏时跳过即可，避免拖垮主链路。
                continue
            if isinstance(item, dict):
                events.append(item)
    return events


def _read_sqlite_events(path: Path) -> list[dict[str, Any]]:
    """读取 SQLite 中的全部 history。

    这个函数主要给 inspect/迁移/清理脚本使用。主链路按 session 查询时
    不走全量读取，而是使用 _filter_session_events 里的索引查询。
    """

    with _connect_sqlite(path) as conn:
        rows = conn.execute(
            "SELECT * FROM conversation_history ORDER BY timestamp ASC, id ASC"
        ).fetchall()
    return [_row_to_event(row) for row in rows]


def read_history_events(
    history_path: str | None = None,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    """读取指定后端中的全部合法会话流水。

    注意：这是治理工具接口，不建议主链路高频调用。
    主链路 summary 读取应使用 get_recent_history/get_all_history。
    """

    resolved_backend = resolve_history_backend(backend, history_path)
    path = resolve_history_path(resolved_backend, history_path)
    if resolved_backend == HISTORY_BACKEND_JSONL:
        return _read_jsonl_events(path)
    return _read_sqlite_events(path)


def _write_jsonl_events(events: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")


def _insert_sqlite_event(conn: sqlite3.Connection, event: dict[str, Any]) -> None:
    """插入一条 SQLite history event。

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
            _json_dumps(_normalize_routes(event.get("routes", []))),
            event.get("user_message", ""),
            event.get("rewritten_query", ""),
            event.get("normalized_question")
            or normalize_history_question(
                event.get("user_message", ""),
                event.get("rewritten_query", ""),
            ),
            event.get("answer_preview", ""),
            _json_dumps(event.get("tags", [])),
            1 if event.get("stored_to_vector") else 0,
            1 if event.get("skipped_vector_store") else 0,
            event.get("vector_store_skip_reason", ""),
            1 if event.get("skipped_duplicate") else 0,
        ),
    )


def _write_sqlite_events(events: list[dict[str, Any]], path: Path) -> None:
    """重写 SQLite history。

    用于 clear/migrate 这类治理脚本。with sqlite3 connection 会形成事务：
    正常退出自动 commit，异常时自动 rollback，避免只写入一半。
    """

    with _connect_sqlite(path) as conn:
        conn.execute("DELETE FROM conversation_history")
        for event in events:
            _insert_sqlite_event(conn, event)


def write_history_events(
    events: list[dict[str, Any]],
    history_path: str | None = None,
    backend: str | None = None,
) -> None:
    """用给定事件列表重写 history。

    主要给清理/治理脚本使用。主链路仍然只做 append，避免额外复杂度。
    """

    resolved_backend = resolve_history_backend(backend, history_path)
    path = resolve_history_path(resolved_backend, history_path)
    if resolved_backend == HISTORY_BACKEND_JSONL:
        _write_jsonl_events(events, path)
        return
    _write_sqlite_events(events, path)


def is_duplicate_history_event(
    *,
    events: list[dict[str, Any]],
    session_id: str,
    user_message: str,
    rewritten_query: str = "",
    routes: list[str] | None = None,
    now_timestamp: float | None = None,
    window_seconds: int | None = None,
) -> bool:
    """判断指定时间窗口内是否已经写过同一条会话事件。

    去重只看“同一 session 下同一路由的同一个标准化问题”，不比较答案内容。
    这样可以抑制重复验证产生的噪音，又不会吞掉不同问题或不同 agent 的记录。
    """

    effective_window = (
        CONVERSATION_HISTORY_CONFIG.dedupe_window_seconds
        if window_seconds is None
        else window_seconds
    )
    if effective_window <= 0:
        return False

    current_time = now_timestamp_s() if now_timestamp is None else now_timestamp
    question = normalize_history_question(user_message, rewritten_query)
    normalized_routes = _normalize_routes(routes)

    for event in reversed(events):
        if event.get("session_id") != session_id:
            continue
        event_timestamp = float(event.get("timestamp") or 0)
        if current_time - event_timestamp > effective_window:
            break

        event_question = normalize_history_question(
            str(event.get("user_message") or ""),
            str(event.get("rewritten_query") or ""),
        )
        if event_question != question:
            continue
        if _normalize_routes(event.get("routes") or []) != normalized_routes:
            continue
        return True

    return False


def _is_duplicate_sqlite_event(
    *,
    conn: sqlite3.Connection,
    event: dict[str, Any],
    window_seconds: int,
) -> bool:
    """用 SQLite 索引判断是否重复。

    查询条件对应 idx_history_dedupe：
    session_id + normalized_question + routes_json + timestamp

    这比 JSONL 版本的“读全文件再倒序扫描”更适合长期运行。
    """

    if window_seconds <= 0:
        return False

    routes_json = _json_dumps(_normalize_routes(event.get("routes", [])))
    threshold = float(event["timestamp"]) - window_seconds
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


def _append_jsonl_event(
    event: dict[str, Any],
    path: Path,
    dedupe: bool,
) -> dict[str, Any]:
    """JSONL 后端追加写。

    JSONL 保留用于兼容旧文件和导出调试；主存储默认已经切到 SQLite。
    """

    existing_events = _read_jsonl_events(path) if dedupe else []
    if dedupe and is_duplicate_history_event(
        events=existing_events,
        session_id=event["session_id"],
        user_message=event["user_message"],
        rewritten_query=event["rewritten_query"],
        routes=event["routes"],
        now_timestamp=event["timestamp"],
    ):
        return {
            "session_id": event["session_id"],
            "user_message": event["user_message"],
            "rewritten_query": event["rewritten_query"],
            "skipped_duplicate": True,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def _append_sqlite_event(
    event: dict[str, Any],
    path: Path,
    dedupe: bool,
) -> dict[str, Any]:
    """SQLite 后端追加写。

    写入前先用索引做短窗口去重；如果不是重复事件，再插入表中。
    这里每次请求使用短连接，代码简单且适合当前单机开发场景。
    """

    with _connect_sqlite(path) as conn:
        if dedupe and _is_duplicate_sqlite_event(
            conn=conn,
            event=event,
            window_seconds=CONVERSATION_HISTORY_CONFIG.dedupe_window_seconds,
        ):
            return {
                "session_id": event["session_id"],
                "user_message": event["user_message"],
                "rewritten_query": event["rewritten_query"],
                "skipped_duplicate": True,
            }
        _insert_sqlite_event(conn, event)
    return event


def append_history_event(
    *,
    session_id: str,
    user_message: str,
    answer: str,
    rewritten_query: str = "",
    routes: list[str] | None = None,
    tags: list[str] | None = None,
    stored_to_vector: bool = False,
    skipped_vector_store: bool = False,
    vector_store_skip_reason: str = "",
    source: str = DEFAULT_HISTORY_SOURCE,
    history_path: str | None = None,
    backend: str | None = None,
    dedupe: bool = DEFAULT_DEDUPE_ENABLED,
) -> dict[str, Any]:
    """追加一条非向量化会话流水。

    这里保存的是“发生过什么”，不是“用于语义召回的知识”。
    因此它适合支撑总结/回放，不适合直接参与 RAG 检索排序。
    """

    resolved_backend = resolve_history_backend(backend, history_path)
    path = resolve_history_path(resolved_backend, history_path)
    event = build_history_event(
        session_id=session_id,
        user_message=user_message,
        answer=answer,
        rewritten_query=rewritten_query,
        routes=routes,
        tags=tags,
        stored_to_vector=stored_to_vector,
        skipped_vector_store=skipped_vector_store,
        vector_store_skip_reason=vector_store_skip_reason,
        source=source,
    )
    if resolved_backend == HISTORY_BACKEND_JSONL:
        return _append_jsonl_event(event, path, dedupe)
    return _append_sqlite_event(event, path, dedupe)


def _filter_session_events(
    *,
    session_id: str,
    limit: int,
    history_path: str | None,
    backend: str | None,
) -> list[dict[str, Any]]:
    """按 session 读取最近 N 条 history。

    SQLite 分支使用 idx_history_session_time 索引：
    WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?

    SQL 里先倒序取最近 N 条性能更好；返回前再 reversed，恢复用户对话的
    正向时间顺序，方便 summary 直接按顺序列出问题。
    """

    resolved_backend = resolve_history_backend(backend, history_path)
    path = resolve_history_path(resolved_backend, history_path)
    if resolved_backend == HISTORY_BACKEND_SQLITE:
        with _connect_sqlite(path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM conversation_history
                WHERE session_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (session_id, max(limit, 0)),
            ).fetchall()
        return list(reversed([_row_to_event(row) for row in rows]))

    events = [
        event
        for event in _read_jsonl_events(path)
        if event.get("session_id") == session_id
    ]
    events = sorted(events, key=lambda item: item.get("timestamp", 0))
    return events[-max(limit, 0) :]


def get_recent_history(
    session_id: str,
    limit: int | None = None,
    history_path: str | None = None,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    """按时间顺序返回当前 session 最近几条会话流水。"""

    effective_limit = limit or CONVERSATION_HISTORY_CONFIG.recent_limit
    return _filter_session_events(
        session_id=session_id,
        limit=effective_limit,
        history_path=history_path,
        backend=backend,
    )


def get_all_history(
    session_id: str,
    limit: int | None = None,
    history_path: str | None = None,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    """按时间顺序返回当前 session 的历史流水，默认只取最近一段窗口。"""

    effective_limit = limit or CONVERSATION_HISTORY_CONFIG.all_limit
    return _filter_session_events(
        session_id=session_id,
        limit=effective_limit,
        history_path=history_path,
        backend=backend,
    )


def preview_history_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "preview": preview(
                event.get("rewritten_query") or event.get("user_message", ""),
                HISTORY_PREVIEW_CHARS,
            ),
            "timestamp": event.get("timestamp"),
            "routes": event.get("routes", []),
            "tags": event.get("tags", []),
            "stored_to_vector": event.get("stored_to_vector", False),
            "vector_store_skip_reason": event.get("vector_store_skip_reason", ""),
        }
        for event in events
    ]

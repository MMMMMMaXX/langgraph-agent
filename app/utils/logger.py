from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from typing import Any

LOGGER_NAME = "langgraph_agent"
DEFAULT_LOG_LEVEL = "INFO"
LOG_TIME_SUFFIX_UTC = "Z"
STRING_PREVIEW_CHARS = 200
WARNING_PREVIEW_CHARS = 240
SUMMARY_PREVIEW_CHARS = 160


class JsonLineFormatter(logging.Formatter):
    """把日志格式化成一行 JSON，方便 stdout 采集器和日志系统解析。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = record.msg if isinstance(record.msg, dict) else {}
        if not payload:
            payload = {"message": record.getMessage()}

        event = {
            "ts": datetime.fromtimestamp(record.created, UTC)
            .isoformat()
            .replace("+00:00", LOG_TIME_SUFFIX_UTC),
            "level": record.levelname,
            "logger": record.name,
            **payload,
        }
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event, ensure_ascii=False, default=str)


def get_app_logger() -> logging.Logger:
    """返回应用统一 logger。

    这里直接写 stdout，是为了让 uvicorn、容器日志和本地终端看到同一份结构化日志。
    """

    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonLineFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    level_name = os.getenv("APP_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    return logger


logger = get_app_logger()


def preview(text: str, n: int = 80) -> str:
    text = str(text).replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[:n] + "..."


def preview_hits(
    hits: list[dict], text_key: str = "content", n: int = 60
) -> list[dict]:
    result = []

    for hit in hits:
        item = {
            "preview": preview(hit.get(text_key, ""), n),
        }

        if "score" in hit:
            try:
                item["score"] = round(float(hit["score"]), 4)
            except (TypeError, ValueError):
                item["score"] = hit["score"]

        if "timestamp" in hit:
            item["timestamp"] = hit["timestamp"]

        if "source" in hit:
            item["source"] = hit["source"]

        if "tags" in hit:
            item["tags"] = hit["tags"]

        result.append(item)

    return result


def compact_summary(summary: str) -> str:
    if not summary:
        return ""
    return preview(summary, SUMMARY_PREVIEW_CHARS)


def is_empty_log_value(value: Any) -> bool:
    """判断字段是否值得输出，减少日志里的空噪音。"""

    return value in (None, "", [], {})


def sanitize_log_value(value: Any, string_limit: int = STRING_PREVIEW_CHARS) -> Any:
    """把日志字段整理成 JSON 友好的值。

    字符串字段做预览截断，复杂对象交给 json.dumps(default=str) 兜底序列化。
    """

    if isinstance(value, str):
        return preview(value, string_limit)
    return value


def sanitize_extra(
    extra: dict[str, Any] | None,
    string_limit: int = STRING_PREVIEW_CHARS,
) -> dict[str, Any]:
    if not extra:
        return {}
    return {
        key: sanitize_log_value(value, string_limit)
        for key, value in extra.items()
        if not is_empty_log_value(value)
    }


def log_node(
    name: str, state: dict[str, Any], extra: dict[str, Any] | None = None
) -> None:
    event: dict[str, Any] = {
        "event": "node",
        "node": name,
    }

    if state.get("request_id"):
        event["request_id"] = state.get("request_id")
    if state.get("session_id"):
        event["session_id"] = state.get("session_id")

    messages = state.get("messages", [])
    if messages:
        latest = messages[-1].get("content", "")
        event["latest_message"] = preview(latest, STRING_PREVIEW_CHARS)

    state_fields = [
        "intent",
        "routes",
        "rewritten_query",
        "answer",
        "summary",
    ]

    for key in state_fields:
        if key not in state:
            continue

        value = state.get(key)

        if is_empty_log_value(value):
            continue

        if key in {"answer", "rewritten_query"}:
            event[key] = preview(str(value), SUMMARY_PREVIEW_CHARS)
        elif key == "summary":
            event[key] = compact_summary(str(value))
        else:
            event[key] = value

    extra_fields = sanitize_extra(extra)
    if extra_fields:
        event["extra"] = extra_fields

    logger.info(event)


def log_request(
    stage: str,
    request_id: str,
    session_id: str,
    message: str = "",
    routes: list[str] | None = None,
    node_timings: dict[str, float] | None = None,
    answer: str = "",
    duration_ms: float | None = None,
    error: str = "",
) -> None:
    event: dict[str, Any] = {
        "event": "request",
        "stage": stage,
        "request_id": request_id,
        "session_id": session_id,
    }

    if message:
        event["message"] = preview(message, STRING_PREVIEW_CHARS)

    if routes:
        event["routes"] = routes

    if node_timings:
        event["node_timings_ms"] = node_timings

    if answer:
        event["answer_preview"] = preview(answer, STRING_PREVIEW_CHARS)

    if duration_ms is not None:
        event["duration_ms"] = round(duration_ms, 2)

    if error:
        event["error"] = preview(error, STRING_PREVIEW_CHARS)

    level = logging.ERROR if error else logging.INFO
    logger.log(level, event)


def now_ms() -> float:
    """返回高精度**耗时测量**用的毫秒数。

    基于 `time.perf_counter()`，单调递增，不受系统时钟调整影响。

    ⚠️ 只用于计算 duration（`now_ms() - started_at_ms`），
    不是墙钟时间戳，不能持久化到 DB、不能跨进程比较。
    需要"当前时间戳"请使用 `now_timestamp_s()`。
    """

    return time.perf_counter() * 1000


def now_timestamp_s() -> float:
    """返回可持久化的**墙钟时间戳**（Unix epoch 秒）。

    基于 `time.time()`，用于写入 DB、向量库、SQLite 等需要跨进程/跨请求
    比较时间先后的场景。

    ⚠️ 不要用它计算耗时（系统时钟可能回跳），测量 duration 请用 `now_ms()`。
    """

    return time.time()


def log_node_timing(
    name: str,
    duration_ms: float,
    request_id: str = "",
    session_id: str = "",
) -> None:
    event: dict[str, Any] = {
        "event": "node_timing",
        "node": name,
        "duration_ms": round(duration_ms, 2),
    }
    if request_id:
        event["request_id"] = request_id
    if session_id:
        event["session_id"] = session_id
    logger.info(event)


def log_warning(
    stage: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """输出非致命降级日志。

    用于“允许 fallback，但不能静默吞掉”的场景，例如 rerank 失败、
    工具参数解析失败、路由 JSON 解析失败。
    """

    event: dict[str, Any] = {
        "event": "warning",
        "stage": stage,
        "message": preview(message, WARNING_PREVIEW_CHARS),
    }
    extra_fields = sanitize_extra(extra, WARNING_PREVIEW_CHARS)
    if extra_fields:
        event["extra"] = extra_fields
    logger.warning(event)

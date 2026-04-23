"""查看非向量化会话流水。

常用示例：
- 查看默认 history 最近 10 条：
  PYTHONPATH=. ./.venv/bin/python scripts/inspect_conversation_history.py --limit 10
- 查看某个 session：
  PYTHONPATH=. ./.venv/bin/python scripts/inspect_conversation_history.py --session-id u1
- 查看某次 eval 的隔离 SQLite history：
  PYTHONPATH=. ./.venv/bin/python scripts/inspect_conversation_history.py --path outputs/eval_runs/xxx.conversation_history.sqlite3 --json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any

from app.config import CONVERSATION_HISTORY_CONFIG
from app.memory.conversation_history import read_history_events, resolve_history_backend
from app.utils.logger import preview

DEFAULT_LIMIT = 20
DEFAULT_PREVIEW_CHARS = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查看 conversation history。")
    parser.add_argument(
        "--backend",
        choices=["sqlite", "jsonl"],
        help="history 后端；不传时根据路径后缀或配置自动判断。",
    )
    parser.add_argument(
        "--path",
        default=CONVERSATION_HISTORY_CONFIG.path,
        help=f"history 路径，默认 {CONVERSATION_HISTORY_CONFIG.path}",
    )
    parser.add_argument("--session-id", help="只展示指定 session_id。")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"最多展示多少条，默认 {DEFAULT_LIMIT}。",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=DEFAULT_PREVIEW_CHARS,
        help=f"预览长度，默认 {DEFAULT_PREVIEW_CHARS}。",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出。")
    return parser.parse_args()


def format_timestamp(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value)).isoformat(timespec="seconds")
    except Exception:
        return ""


def filter_events(events: list[dict[str, Any]], session_id: str | None) -> list[dict[str, Any]]:
    if not session_id:
        return events
    return [event for event in events if event.get("session_id") == session_id]


def build_output(events: list[dict[str, Any]], preview_chars: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for event in events:
        output.append(
            {
                "session_id": event.get("session_id", ""),
                "timestamp": event.get("timestamp"),
                "time": format_timestamp(event.get("timestamp")),
                "routes": event.get("routes", []),
                "question": event.get("rewritten_query") or event.get("user_message", ""),
                "answer_preview": preview(event.get("answer_preview", ""), preview_chars),
                "tags": event.get("tags", []),
                "stored_to_vector": event.get("stored_to_vector", False),
                "vector_store_skip_reason": event.get("vector_store_skip_reason", ""),
            }
        )
    return output


def print_text_output(
    *,
    backend: str,
    path: str,
    total_count: int,
    records: list[dict[str, Any]],
) -> None:
    print(f"backend: {backend}")
    print(f"path: {path}")
    print(f"total_count: {total_count}")
    print(f"returned_count: {len(records)}")

    for index, record in enumerate(records, start=1):
        print(f"\n--- event {index} ---")
        print("time:", record["time"])
        print("session_id:", record["session_id"])
        print("routes:", record["routes"])
        print("question:", record["question"])
        print("tags:", record["tags"])
        print("stored_to_vector:", record["stored_to_vector"])
        print("vector_store_skip_reason:", record["vector_store_skip_reason"])
        print("answer_preview:", record["answer_preview"])


def main() -> None:
    args = parse_args()
    backend = resolve_history_backend(args.backend, args.path)
    events = read_history_events(args.path, backend=backend)
    filtered = filter_events(events, args.session_id)
    filtered = sorted(filtered, key=lambda item: item.get("timestamp", 0))
    limited = filtered[-max(args.limit, 0) :]
    records = build_output(limited, args.preview_chars)

    if args.json:
        print(
            json.dumps(
                {
                    "path": args.path,
                    "backend": backend,
                    "total_count": len(events),
                    "filtered_count": len(filtered),
                    "returned_count": len(records),
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print_text_output(
        backend=backend,
        path=args.path,
        total_count=len(events),
        records=records,
    )


if __name__ == "__main__":
    main()

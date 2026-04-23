"""清理非向量化会话流水。

默认只 dry-run，不会真正改文件；确认后加 --apply。

常用示例：
- 预览清理某个 session：
  PYTHONPATH=. ./.venv/bin/python scripts/clear_conversation_history.py --session-id u1
- 真正清理某个 session：
  PYTHONPATH=. ./.venv/bin/python scripts/clear_conversation_history.py --session-id u1 --apply
- 清空某个 eval SQLite history 文件：
  PYTHONPATH=. ./.venv/bin/python scripts/clear_conversation_history.py --path outputs/eval_runs/xxx.conversation_history.sqlite3 --all --apply
"""

from __future__ import annotations

import argparse

from app.config import CONVERSATION_HISTORY_CONFIG
from app.memory.conversation_history import (
    read_history_events,
    resolve_history_backend,
    write_history_events,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清理 conversation history。")
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
    parser.add_argument("--session-id", help="清理指定 session_id 的记录。")
    parser.add_argument("--all", action="store_true", help="清空整个 history 文件。")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正写回文件；不加时只预览影响范围。",
    )
    return parser.parse_args()


def should_remove_event(event: dict, session_id: str | None, remove_all: bool) -> bool:
    if remove_all:
        return True
    return bool(session_id) and event.get("session_id") == session_id


def main() -> None:
    args = parse_args()
    if not args.all and not args.session_id:
        raise SystemExit("请指定 --session-id 或 --all。默认不会盲目清理。")

    backend = resolve_history_backend(args.backend, args.path)
    events = read_history_events(args.path, backend=backend)
    kept_events = [
        event
        for event in events
        if not should_remove_event(event, args.session_id, args.all)
    ]
    removed_count = len(events) - len(kept_events)

    print(f"backend: {backend}")
    print(f"path: {args.path}")
    print(f"total_count: {len(events)}")
    print(f"removed_count: {removed_count}")
    print(f"kept_count: {len(kept_events)}")

    if not args.apply:
        print("dry_run: true")
        print("提示：确认无误后加 --apply 才会写回文件。")
        return

    write_history_events(kept_events, args.path, backend=backend)
    print("applied: true")


if __name__ == "__main__":
    main()

"""把旧 JSONL conversation history 迁移到 SQLite。

默认 dry-run，不会写入；确认后加 --apply。

示例：
  PYTHONPATH=. ./.venv/bin/python scripts/migrate_conversation_history_jsonl_to_sqlite.py
  PYTHONPATH=. ./.venv/bin/python scripts/migrate_conversation_history_jsonl_to_sqlite.py --apply
"""

from __future__ import annotations

import argparse

from app.config import CONVERSATION_HISTORY_CONFIG
from app.memory.conversation_history import read_history_events, write_history_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 JSONL conversation history 迁移到 SQLite。"
    )
    parser.add_argument(
        "--source",
        default=CONVERSATION_HISTORY_CONFIG.jsonl_path,
        help=f"源 JSONL 路径，默认 {CONVERSATION_HISTORY_CONFIG.jsonl_path}",
    )
    parser.add_argument(
        "--target",
        default=CONVERSATION_HISTORY_CONFIG.sqlite_path,
        help=f"目标 SQLite 路径，默认 {CONVERSATION_HISTORY_CONFIG.sqlite_path}",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正写入 SQLite；不加时只预览迁移数量。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = read_history_events(args.source, backend="jsonl")

    print(f"source: {args.source}")
    print(f"target: {args.target}")
    print(f"event_count: {len(events)}")

    if not args.apply:
        print("dry_run: true")
        print("提示：确认无误后加 --apply 才会写入 SQLite。")
        return

    write_history_events(events, args.target, backend="sqlite")
    print("applied: true")


if __name__ == "__main__":
    main()

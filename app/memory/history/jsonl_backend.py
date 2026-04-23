"""JSONL 后端实现。

保留用途：
- 兼容历史的 .jsonl 文件，允许老数据不迁移就能继续读
- 导出/调试时人眼可直接查看
- eval 场景里临时落盘小样本

不适合长期高频写，因为每次 append（在启用 dedupe 时）都要先把
整份文件读进内存再判重，O(N) 扫描不如 SQLite 索引。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .events import is_duplicate_in_memory, make_dedupe_stub


class JsonlBackend:
    """行分隔 JSON 的 HistoryBackend 实现。"""

    def __init__(self, dedupe_window_seconds: int) -> None:
        self.dedupe_window_seconds = dedupe_window_seconds

    # ---------- low-level file io ----------

    @staticmethod
    def _read_all_lines(path: Path) -> list[dict[str, Any]]:
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

    @staticmethod
    def _write_all_lines(events: list[dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for event in events:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")

    # ---------- HistoryBackend interface ----------

    def append(
        self,
        event: dict[str, Any],
        path: Path,
        dedupe: bool,
    ) -> dict[str, Any]:
        """追加一条 event，启用 dedupe 时先把文件读进来判断。"""

        existing_events = self._read_all_lines(path) if dedupe else []
        if dedupe and is_duplicate_in_memory(
            events=existing_events,
            session_id=event["session_id"],
            user_message=event["user_message"],
            rewritten_query=event["rewritten_query"],
            routes=event["routes"],
            now_timestamp=event["timestamp"],
            window_seconds=self.dedupe_window_seconds,
        ):
            return make_dedupe_stub(event)

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def read_session(
        self,
        session_id: str,
        limit: int,
        path: Path,
    ) -> list[dict[str, Any]]:
        """全量读取后按 session 过滤，再取最近 limit 条。

        JSONL 没有索引，这里只能线性扫描；仅兜底场景使用，不要走主链路高频路径。
        """

        capped = max(limit, 0)
        if capped == 0:
            return []
        events = [
            event
            for event in self._read_all_lines(path)
            if event.get("session_id") == session_id
        ]
        events.sort(key=lambda item: item.get("timestamp", 0))
        return events[-capped:]

    def read_all(self, path: Path) -> list[dict[str, Any]]:
        return self._read_all_lines(path)

    def write_all(self, events: list[dict[str, Any]], path: Path) -> None:
        self._write_all_lines(events, path)

"""HistoryBackend Protocol：所有存储后端必须满足的结构化接口。

使用 typing.Protocol 而不是 abc.ABC：
- 让后端实现无需继承基类，只要方法签名匹配即可被 service.py 当作 backend
- 方便 eval / 测试里传入内存版 stub，不需要构造真实文件
- 避免多重继承的层级问题

所有方法都应该是幂等或显式写入语义，具体约束见各方法 docstring。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class HistoryBackend(Protocol):
    """history 存储后端的最小行为集合。"""

    dedupe_window_seconds: int
    """
    去重窗口（秒）。当调用方请求 dedupe=True 时，后端会在这个时间窗口内
    判断是否已经有同样的 session+question+routes 记录。
    <= 0 表示关闭去重。

    按约定，window 值由 service 层在构造 backend 时注入，后端自身
    不应该再去读全局 CONVERSATION_HISTORY_CONFIG，避免层级依赖倒挂。
    """

    def append(
        self,
        event: dict[str, Any],
        path: Path,
        dedupe: bool,
    ) -> dict[str, Any]:
        """追加写一条 event。

        dedupe 为 True 时，如果窗口内已有重复记录，返回 make_dedupe_stub(event)，
        不应实际写入。否则写入成功，返回原 event。
        """

    def read_session(
        self,
        session_id: str,
        limit: int,
        path: Path,
    ) -> list[dict[str, Any]]:
        """按时间升序返回指定 session 的最近 `limit` 条事件。

        SQLite 实现应使用 (session_id, timestamp) 索引，避免全表扫描。
        limit <= 0 时返回空列表。
        """

    def read_all(self, path: Path) -> list[dict[str, Any]]:
        """读取全部事件，按时间升序返回。

        给治理脚本（inspect / migrate / clear）使用，主链路不要调用。
        """

    def write_all(self, events: list[dict[str, Any]], path: Path) -> None:
        """用给定列表完整覆盖 history 存储。

        用于清理/迁移脚本，主链路不调用。
        """

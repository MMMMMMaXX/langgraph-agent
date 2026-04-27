from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.constants.runtime import RUNTIME_RESTORE_FROM_EMPTY


@dataclass(slots=True)
class ConversationSnapshot:
    """统一表示某个 session 当前可恢复的会话快照。

    这个对象只表达“本轮 invoke 开始前我们手里有什么状态”，不负责持久化行为。
    后续无论底层来自 session cache、checkpoint，还是将来来自外部 session store，
    上层都只需要消费这一份统一结构。
    """

    session_id: str
    thread_id: str
    messages: list[Any] = field(default_factory=list)
    summary: str = ""
    restored_from: str = RUNTIME_RESTORE_FROM_EMPTY
    has_checkpoint_state: bool = False

    @property
    def message_count(self) -> int:
        """返回当前快照中的消息数量，便于 debug / trace 打点。"""

        return len(self.messages)

    @property
    def is_empty(self) -> bool:
        """判断该快照是否完全为空。"""

        return not self.messages and not self.summary

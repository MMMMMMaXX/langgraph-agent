"""兼容门面：保留旧导入路径。

当前真正实现已经迁到 `app.memory.services` 目录。
保留这个模块，是为了避免历史代码或临时脚本仍引用旧路径时直接报错。
"""

from app.memory.services import (
    HistoryStoreResult,
    MessagePruneResult,
    SummaryRefreshResult,
    VectorStoreResult,
    build_memory_debug_payload,
    build_memory_log_extra,
    prune_working_messages,
    refresh_summary_if_needed,
    write_history_if_needed,
    write_vector_memory_if_needed,
)

__all__ = [
    "SummaryRefreshResult",
    "VectorStoreResult",
    "HistoryStoreResult",
    "MessagePruneResult",
    "build_memory_debug_payload",
    "build_memory_log_extra",
    "refresh_summary_if_needed",
    "write_vector_memory_if_needed",
    "write_history_if_needed",
    "prune_working_messages",
]

"""memory node 内部服务入口。

按主题拆分 summary / history / vector / prune / debug payload，避免单个
node_services.py 继续膨胀。
"""

from .debug_payloads import build_memory_debug_payload, build_memory_log_extra
from .history_write_service import HistoryStoreResult, write_history_if_needed
from .prune_service import MessagePruneResult, prune_working_messages
from .summary_service import SummaryRefreshResult, refresh_summary_if_needed
from .vector_write_service import VectorStoreResult, write_vector_memory_if_needed

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

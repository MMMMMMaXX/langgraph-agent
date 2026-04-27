from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

ErrorBuilder = Callable[..., dict]
PreviewFn = Callable[[str, int], str]


@dataclass(slots=True)
class SummaryRefreshResult:
    summary: str
    refreshed_summary: bool
    skipped_summary_refresh: bool
    errors: list[dict] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass(slots=True)
class VectorStoreResult:
    stored_to_vector: bool
    skipped_vector_store: bool
    vector_store_skip_reason: str
    stored_tags: list[str] = field(default_factory=list)
    stored_preview: str = ""
    errors: list[dict] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass(slots=True)
class HistoryStoreResult:
    stored_to_history: bool
    skipped_history_store: bool
    history_store_skip_reason: str
    history_preview: str = ""
    errors: list[dict] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass(slots=True)
class MessagePruneResult:
    messages: list[dict]
    duration_ms: float = 0.0

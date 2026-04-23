from dataclasses import dataclass, field

from app.constants.policies import HISTORY_POLICY_NONE


@dataclass
class ChatAnswerResult:
    """chat agent 回答阶段的统一结果。"""

    answer: str
    history_lookup_policy: str = HISTORY_POLICY_NONE
    working_memory_items: list[str] = field(default_factory=list)
    history_events: list[dict] = field(default_factory=list)
    history_items: list[str] = field(default_factory=list)
    used_memory: bool = False
    used_history: bool = False
    used_summary: bool = False
    errors: list[str] = field(default_factory=list)
    sub_timings_ms: dict[str, float] = field(default_factory=dict)

"""Phase 3 Multi-hop RAG 子包。

模块职责分工（PR-1 只落地 types / gate / decompose）：
- `types.py`    : EvidencePreview / EvidenceGroup / Subquery / DecomposeResult 等纯数据类型
- `gate.py`     : `should_enter_multi_hop(query)` —— negative gate + 正向触发正则
- `decompose.py`: LLM decompose + JSON schema 校验 + 降级判定

`gap.py` / `node.py` 由后续 PR 补；本子包不导入 app/state、LangGraph 相关依赖，
保持纯函数可单测。
"""

from app.agents.rag.multi_hop.gate import (
    preprocess_query_for_gate,
    should_enter_multi_hop,
)
from app.agents.rag.multi_hop.types import (
    DecomposeResult,
    EvidenceGroup,
    EvidencePreview,
    Subquery,
)

__all__ = [
    "DecomposeResult",
    "EvidenceGroup",
    "EvidencePreview",
    "Subquery",
    "preprocess_query_for_gate",
    "should_enter_multi_hop",
]

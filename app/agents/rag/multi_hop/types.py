"""Multi-hop RAG 内部数据类型。

设计原则：
- **EvidencePreview 不存全文**：只保留 <=120 chars 预览 + 元数据，保证 AgentState /
  checkpoint / LangSmith trace 不泄漏原文。node 内部另持 `full_chunks_for_answer`
  局部 dict 送 answer LLM（见 `docs/phase3-multi-hop-rag.md` §2.2）。
- **Subquery.depends_on 用 tuple**：state 需序列化进 checkpoint，不可变类型更安全。
- **DecomposeResult.degraded_to_single_hop**：Decomposer 主动宣告"我拆不动了"，
  由 multi_hop_node 走 fallback_to_single_hop 分支；不靠下游反推。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Subquery:
    """Decomposer 产出的单个子查询。"""

    id: str
    intent: str
    query: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecomposeResult:
    """Decomposer 的最终输出。

    - `degraded_to_single_hop=True` 时 subqueries 仍保留一个"兜底"子查询
      （值等于 rewritten_query），供 node 直接走单跳 fallback。
    - `error_code` 为空表示正常；否则由 node 写入 debug_info。
    """

    subqueries: tuple[Subquery, ...]
    degraded_to_single_hop: bool
    reason: str = ""
    error_code: str = ""


@dataclass(frozen=True)
class EvidencePreview:
    """进入 state / trace 的最小证据形态；全文不进入这里。"""

    doc_id: str
    chunk_id: str
    ref: str
    score: float
    preview: str


@dataclass(frozen=True)
class EvidenceGroup:
    """单 subquery 的证据聚合 + coverage 元数据。"""

    subquery_id: str
    chunks: tuple[EvidencePreview, ...]
    per_subquery_coverage: float
    missing_aspects: tuple[str, ...] = ()
    hop: int = 0


@dataclass
class MultiHopDebugInfo:
    """写入 `debug_info.multi_hop` 的可选分层指标。

    使用可变 dataclass 便于 node 增量填充；最终由 node 转成 dict 塞进 state。
    """

    decompose_ms: float = 0.0
    retrieval_ms_per_subquery: list[float] = field(default_factory=list)
    per_subquery_coverage: dict[str, float] = field(default_factory=dict)
    global_coverage: float = 0.0
    degrade_reason: str = ""
    hop_count: int = 0
    total_llm_calls: int = 0
    total_embedding_calls: int = 0


__all__ = [
    "DecomposeResult",
    "EvidenceGroup",
    "EvidencePreview",
    "MultiHopDebugInfo",
    "Subquery",
]

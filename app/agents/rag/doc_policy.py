"""RAG 文档 rerank 策略。

这里放的是“是否需要 LLM rerank”的判断规则。它只读取候选文档命中的
元信息，不做检索、不改写候选内容，方便单独测试延迟优化策略。
"""

from app.agents.rag.constants import (
    DOC_RERANK_SKIP_MAX_CANDIDATES,
    DOC_RERANK_SKIP_SCORE_DELTA,
    HIGH_CONFIDENCE_RETRIEVAL_SOURCES,
)
from app.constants.policies import (
    DOC_RERANK_SKIP_REASON_ADJACENT_HIGH_CONFIDENCE,
    DOC_RERANK_SKIP_REASON_SINGLE_CANDIDATE,
)


def has_high_confidence_sources(hit: dict) -> bool:
    """判断命中是否同时来自高置信召回支路。"""

    sources = set(hit.get("retrieval_sources") or [])
    return HIGH_CONFIDENCE_RETRIEVAL_SOURCES.issubset(sources)


def are_same_doc_adjacent_hits(hits: list[dict]) -> bool:
    """判断候选是否来自同一文档的相邻 chunk。"""

    if len(hits) < 2:
        return False

    doc_ids = {hit.get("doc_id") for hit in hits}
    if len(doc_ids) != 1:
        return False

    chunk_indexes = sorted(hit.get("chunk_index", 0) for hit in hits)
    return all(
        current + 1 == next_index
        for current, next_index in zip(chunk_indexes, chunk_indexes[1:], strict=False)
    )


def should_skip_doc_rerank(hits: list[dict]) -> tuple[bool, str]:
    """判断文档候选是否可以跳过 LLM rerank。

    LLM rerank 对少量、高置信、同文档相邻 chunk 的收益通常很低，但会增加
    1~2 秒延迟。这里先做保守规则：只有候选很少、分差很小、都由
    dense+keyword 双路命中，且属于同一文档相邻 chunk 时才跳过。
    """

    if len(hits) <= 1:
        return True, DOC_RERANK_SKIP_REASON_SINGLE_CANDIDATE

    if len(hits) > DOC_RERANK_SKIP_MAX_CANDIDATES:
        return False, ""

    top_score = hits[0].get("score", 0.0)
    bottom_score = hits[-1].get("score", 0.0)
    if abs(top_score - bottom_score) > DOC_RERANK_SKIP_SCORE_DELTA:
        return False, ""

    if not all(has_high_confidence_sources(hit) for hit in hits):
        return False, ""

    if not are_same_doc_adjacent_hits(hits):
        return False, ""

    return True, DOC_RERANK_SKIP_REASON_ADJACENT_HIGH_CONFIDENCE

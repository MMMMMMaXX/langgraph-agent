"""RAG memory 兜底检索管线。"""

from app.constants.keywords import (
    RECALL_QUERY_KEYWORDS,
    SUMMARY_QUERY_KEYWORDS,
    contains_any,
)
from app.agents.rag.types import MemoryRetrievalResult
from app.config import RAG_CONFIG, VECTOR_STORE_CONFIG
from app.constants.policies import SKIP_REASON_DOC_HIT
from app.memory.vector_memory import search_memory
from app.retrieval.reranker import rerank
from app.utils.errors import build_error_info
from app.utils.logger import now_ms
from app.utils.memory_key import dedupe_memory_hits


def filter_memory_hits(memory_hits: list[dict]) -> list[dict]:
    """过滤不适合作为 RAG 兜底证据的 memory 命中。

    RAG fallback 需要的是可回答用户问题的事实型记忆；总结、回顾、刚刚
    这类元问题记录如果进入上下文，容易让模型把“问过什么”当成答案事实。
    """

    filtered = []
    for memory_hit in memory_hits:
        content = memory_hit["content"]

        if contains_any(content, SUMMARY_QUERY_KEYWORDS):
            continue
        if contains_any(content, RECALL_QUERY_KEYWORDS):
            continue

        filtered.append(memory_hit)

    return filtered


def retrieve_memory_for_rag(
    query: str,
    *,
    session_id: str,
    enabled: bool,
) -> MemoryRetrievalResult:
    """在文档命中不足时执行 memory 兜底检索。"""

    timings_ms = {
        "memorySearch": 0.0,
        "memoryRerank": 0.0,
    }
    errors: list[str] = []
    memory_hits: list[dict] = []
    memory_before_rerank: list[dict] = []

    if enabled:
        memory_search_started_at_ms = now_ms()
        try:
            memory_hits = search_memory(
                query,
                top_k=RAG_CONFIG.memory_top_k,
                session_id=session_id,
            )
        except Exception as exc:
            memory_hits = []
            errors.append(
                build_error_info(
                    exc,
                    stage="search_memory",
                    source="memory",
                    preferred_code="retrieval_error",
                )
            )

        # 先去掉“总结/刚刚”这类元问题，再做 key 去重。
        memory_hits = filter_memory_hits(memory_hits)
        memory_hits = dedupe_memory_hits(memory_hits)
        timings_ms["memorySearch"] = round(now_ms() - memory_search_started_at_ms, 2)
        memory_before_rerank = memory_hits[:]

        memory_rerank_started_at_ms = now_ms()
        try:
            memory_hits = rerank(
                query,
                memory_hits,
                top_k=RAG_CONFIG.memory_rerank_top_k,
            )
        except Exception as exc:
            memory_hits = memory_before_rerank[: RAG_CONFIG.memory_rerank_top_k]
            errors.append(
                build_error_info(
                    exc,
                    stage="memory_rerank",
                    source="retrieval",
                    preferred_code="retrieval_error",
                )
            )
        timings_ms["memoryRerank"] = round(now_ms() - memory_rerank_started_at_ms, 2)

    retrieval_debug = {
        "collection": VECTOR_STORE_CONFIG.memory_collection_name,
        "where": {"session_id": session_id} if enabled else None,
        "enabled": enabled,
        "skip_reason": "" if enabled else SKIP_REASON_DOC_HIT,
        "requested_top_k": RAG_CONFIG.memory_top_k,
        "candidate_top_k": max(RAG_CONFIG.memory_top_k * 4, RAG_CONFIG.memory_top_k),
        "returned_count": len(memory_before_rerank),
        "consumed_count": len(memory_hits),
    }

    return MemoryRetrievalResult(
        memory_hits=memory_hits,
        memory_before_rerank=memory_before_rerank,
        retrieval_debug=retrieval_debug,
        errors=errors,
        timings_ms=timings_ms,
    )

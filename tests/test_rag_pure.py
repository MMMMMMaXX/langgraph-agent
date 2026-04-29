"""RAG 纯函数单测：chunk_merge + doc_policy。

这两个模块都是"读 dict 列表 → 返回 dict 列表/布尔"，完全无 I/O。
相比带检索/LLM 的 pipeline，这里可以把分支几乎全覆盖到。
"""

from __future__ import annotations

from app.agents.rag.chunk_merge import merge_adjacent_doc_hits
from app.agents.rag.doc_policy import (
    are_same_doc_adjacent_hits,
    has_high_confidence_sources,
    should_skip_doc_rerank,
)
from app.constants.policies import (
    DOC_RERANK_SKIP_REASON_ADJACENT_HIGH_CONFIDENCE,
    DOC_RERANK_SKIP_REASON_SINGLE_CANDIDATE,
)

# ---------------------------------------------------------------------------
# chunk_merge
# ---------------------------------------------------------------------------


def _hit(
    doc_id: str,
    chunk_index: int,
    score: float = 0.9,
    content: str = "text",
    *,
    hit_id: str | None = None,
    sources: list[str] | None = None,
) -> dict:
    return {
        "id": hit_id or f"{doc_id}-{chunk_index}",
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "start_char": chunk_index * 100,
        "end_char": chunk_index * 100 + len(content),
        "content": content,
        "score": score,
        "semantic_score": score,
        "keyword_score_norm": score,
        "retrieval_sources": sources or ["dense"],
    }


def test_merge_empty_returns_empty() -> None:
    assert merge_adjacent_doc_hits([]) == []


def test_merge_single_hit_preserves_fields_and_adds_merge_metadata() -> None:
    hits = [_hit("d1", 0, content="hello")]
    merged = merge_adjacent_doc_hits(hits)

    assert len(merged) == 1
    result = merged[0]
    assert result["content"] == "hello"
    # 单条也会加上 merge 元数据，方便下游统一处理
    assert result["merged_chunk_ids"] == ["d1-0"]
    assert result["merged_chunk_indexes"] == [0]


def test_merge_two_adjacent_same_doc_are_merged() -> None:
    hits = [
        _hit("d1", 0, score=0.9, content="part one"),
        _hit("d1", 1, score=0.7, content="part two"),
    ]
    merged = merge_adjacent_doc_hits(hits)

    assert len(merged) == 1
    result = merged[0]
    # 内容用分隔符拼接
    assert "part one" in result["content"] and "part two" in result["content"]
    # id 用 "+" 连起来
    assert result["id"] == "d1-0+d1-1"
    # 分数取最大
    assert result["score"] == 0.9
    # 起止位置覆盖整段
    assert result["start_char"] == 0
    assert result["end_char"] > 100


def test_merge_non_adjacent_same_doc_not_merged() -> None:
    """chunk_index 不连续（0 和 2）不应合并。"""
    hits = [
        _hit("d1", 0, content="a"),
        _hit("d1", 2, content="b"),
    ]
    merged = merge_adjacent_doc_hits(hits)
    assert len(merged) == 2


def test_merge_different_docs_not_merged() -> None:
    hits = [
        _hit("d1", 0, content="alpha"),
        _hit("d2", 0, content="beta"),
    ]
    merged = merge_adjacent_doc_hits(hits)
    assert len(merged) == 2


def test_merge_retrieval_sources_deduped_and_sorted() -> None:
    hits = [
        _hit("d1", 0, sources=["dense", "keyword"]),
        _hit("d1", 1, sources=["keyword", "dense"]),
    ]
    merged = merge_adjacent_doc_hits(hits)
    assert merged[0]["retrieval_sources"] == ["dense", "keyword"]


def test_merge_final_sort_by_score_desc() -> None:
    """合并完后按 score 降序排，低分不该排到前面。"""
    hits = [
        _hit("d1", 0, score=0.3, content="low"),
        _hit("d2", 0, score=0.9, content="high"),
    ]
    merged = merge_adjacent_doc_hits(hits)
    assert merged[0]["content"] == "high"
    assert merged[1]["content"] == "low"


def test_merge_skips_empty_content_in_concatenation() -> None:
    """content 为空串的 chunk 不会污染合并结果。"""
    hits = [
        _hit("d1", 0, content="a"),
        _hit("d1", 1, content=""),
        _hit("d1", 2, content="c"),
    ]
    merged = merge_adjacent_doc_hits(hits)
    assert len(merged) == 1
    # 空串被 filter 掉，不应出现双分隔符
    assert "\n\n" not in merged[0]["content"]
    assert "a" in merged[0]["content"] and "c" in merged[0]["content"]


# ---------------------------------------------------------------------------
# doc_policy
# ---------------------------------------------------------------------------


def test_has_high_confidence_requires_both_dense_and_keyword() -> None:
    assert has_high_confidence_sources({"retrieval_sources": ["dense", "keyword"]})
    assert has_high_confidence_sources(
        {"retrieval_sources": ["dense", "keyword", "bm25"]}
    )
    # 缺一个都不行
    assert not has_high_confidence_sources({"retrieval_sources": ["dense"]})
    assert not has_high_confidence_sources({"retrieval_sources": ["keyword"]})
    # 没有 retrieval_sources 字段
    assert not has_high_confidence_sources({})
    # retrieval_sources 为 None（容错）
    assert not has_high_confidence_sources({"retrieval_sources": None})


def test_are_same_doc_adjacent_hits_requires_at_least_two() -> None:
    assert not are_same_doc_adjacent_hits([])
    assert not are_same_doc_adjacent_hits([{"doc_id": "d1", "chunk_index": 0}])


def test_are_same_doc_adjacent_hits_detects_adjacent() -> None:
    hits = [
        {"doc_id": "d1", "chunk_index": 0},
        {"doc_id": "d1", "chunk_index": 1},
    ]
    assert are_same_doc_adjacent_hits(hits)


def test_are_same_doc_adjacent_hits_rejects_non_adjacent() -> None:
    hits = [
        {"doc_id": "d1", "chunk_index": 0},
        {"doc_id": "d1", "chunk_index": 2},  # gap
    ]
    assert not are_same_doc_adjacent_hits(hits)


def test_are_same_doc_adjacent_hits_rejects_different_docs() -> None:
    hits = [
        {"doc_id": "d1", "chunk_index": 0},
        {"doc_id": "d2", "chunk_index": 1},
    ]
    assert not are_same_doc_adjacent_hits(hits)


# should_skip_doc_rerank 综合策略
# 跳过条件：候选 <=1 / <=2 候选 + 分差 <=0.05 + 全部高置信 + 同文档相邻


def test_skip_rerank_single_candidate_always_skips() -> None:
    hits = [_hit("d1", 0)]
    skip, reason = should_skip_doc_rerank(hits)
    assert skip is True
    assert reason == DOC_RERANK_SKIP_REASON_SINGLE_CANDIDATE


def test_skip_rerank_two_adjacent_high_confidence_skips() -> None:
    hits = [
        _hit("d1", 0, score=0.85, sources=["dense", "keyword"]),
        _hit("d1", 1, score=0.83, sources=["dense", "keyword"]),
    ]
    skip, reason = should_skip_doc_rerank(hits)
    assert skip is True
    assert reason == DOC_RERANK_SKIP_REASON_ADJACENT_HIGH_CONFIDENCE


def test_skip_rerank_too_many_candidates_does_not_skip() -> None:
    hits = [_hit("d1", i) for i in range(3)]  # > DOC_RERANK_SKIP_MAX_CANDIDATES=2
    skip, reason = should_skip_doc_rerank(hits)
    assert skip is False
    assert reason == ""


def test_skip_rerank_score_delta_too_large_does_not_skip() -> None:
    hits = [
        _hit("d1", 0, score=0.9, sources=["dense", "keyword"]),
        _hit("d1", 1, score=0.5, sources=["dense", "keyword"]),  # 分差 0.4 > 0.05
    ]
    skip, _ = should_skip_doc_rerank(hits)
    assert skip is False


def test_skip_rerank_missing_high_confidence_does_not_skip() -> None:
    hits = [
        _hit("d1", 0, score=0.85, sources=["dense"]),  # 缺 keyword
        _hit("d1", 1, score=0.83, sources=["dense", "keyword"]),
    ]
    skip, _ = should_skip_doc_rerank(hits)
    assert skip is False


def test_skip_rerank_non_adjacent_does_not_skip() -> None:
    hits = [
        _hit("d1", 0, score=0.85, sources=["dense", "keyword"]),
        _hit("d1", 3, score=0.83, sources=["dense", "keyword"]),  # chunk 非相邻
    ]
    skip, _ = should_skip_doc_rerank(hits)
    assert skip is False

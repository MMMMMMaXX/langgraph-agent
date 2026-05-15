"""Phase 3 PR-2：Gap Detector 单测。

覆盖 §6.1 per_subquery 三条规则、§6.2 global 三条规则、以及 missing_aspect →
结构化 refine 参数映射。所有用例用伪 `EvidencePreview` 注入，不触 retrieval。
"""

from __future__ import annotations

from app.agents.rag.multi_hop.gap import (
    MISSING_CHAIN_BROKEN_PREFIX,
    MISSING_ENTITY_PREFIX,
    MISSING_GLOBAL_ENTITY_PREFIX,
    MISSING_GLOBAL_NO_DIVERSITY,
    MISSING_INSUFFICIENT_CHUNKS,
    MISSING_LOW_CONFIDENCE,
    compute_global_coverage,
    compute_per_subquery_coverage,
    detect_gaps,
)
from app.agents.rag.multi_hop.types import EvidenceGroup, EvidencePreview, Subquery
from app.constants.multi_hop import (
    MIN_CHUNK_SCORE,
    MIN_DOCS_MULTI,
    SUBQUERY_INTENT_COMPARISON_ARM,
    SUBQUERY_INTENT_ENTITY_LOOKUP,
    SUBQUERY_INTENT_PROCEDURE,
)

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _mk_preview(
    *,
    doc_id: str = "d1",
    chunk_id: str = "c1",
    score: float = 0.8,
    preview: str = "some text",
    ref: str = "[1]",
) -> EvidencePreview:
    return EvidencePreview(
        doc_id=doc_id,
        chunk_id=chunk_id,
        ref=ref,
        score=score,
        preview=preview,
    )


def _mk_group(
    subquery_id: str,
    chunks: tuple[EvidencePreview, ...],
) -> EvidenceGroup:
    # per_subquery_coverage 留 0.0，detect_gaps 不依赖这个值——它自己算。
    return EvidenceGroup(
        subquery_id=subquery_id,
        chunks=chunks,
        per_subquery_coverage=0.0,
    )


# ---------------------------------------------------------------------------
# per_subquery 规则
# ---------------------------------------------------------------------------


def test_per_subquery_full_coverage_returns_1() -> None:
    sq = Subquery(id="sq1", intent=SUBQUERY_INTENT_PROCEDURE, query="X")
    chunks = (_mk_preview(score=0.8), _mk_preview(score=0.7, chunk_id="c2"))
    coverage, missing = compute_per_subquery_coverage(subquery=sq, chunks=chunks)
    assert coverage == 1.0
    assert missing == ()


def test_per_subquery_insufficient_chunks_missing() -> None:
    sq = Subquery(id="sq1", intent=SUBQUERY_INTENT_PROCEDURE, query="X")
    chunks = (_mk_preview(score=0.8),)  # < MIN_CHUNKS_PER_SUBQUERY (=2)
    coverage, missing = compute_per_subquery_coverage(subquery=sq, chunks=chunks)
    assert MISSING_INSUFFICIENT_CHUNKS in missing
    assert coverage < 1.0


def test_per_subquery_low_confidence_missing() -> None:
    """chunks 够数但全部低分 → low_confidence。"""

    sq = Subquery(id="sq1", intent=SUBQUERY_INTENT_PROCEDURE, query="X")
    low = MIN_CHUNK_SCORE - 0.1
    chunks = (
        _mk_preview(score=low, chunk_id="c1"),
        _mk_preview(score=low, chunk_id="c2"),
    )
    coverage, missing = compute_per_subquery_coverage(subquery=sq, chunks=chunks)
    assert MISSING_LOW_CONFIDENCE in missing
    assert MISSING_INSUFFICIENT_CHUNKS not in missing
    assert coverage < 1.0


def test_per_subquery_entity_miss_missing() -> None:
    sq = Subquery(id="sq1", intent=SUBQUERY_INTENT_ENTITY_LOOKUP, query="JWT 是什么")
    chunks = (
        _mk_preview(preview="this talks about something else", chunk_id="c1"),
        _mk_preview(preview="irrelevant content", chunk_id="c2"),
    )
    coverage, missing = compute_per_subquery_coverage(
        subquery=sq, chunks=chunks, entity_hints=("JWT",)
    )
    assert f"{MISSING_ENTITY_PREFIX}JWT" in missing
    assert coverage < 1.0


def test_per_subquery_entity_hint_case_insensitive() -> None:
    sq = Subquery(id="sq1", intent=SUBQUERY_INTENT_ENTITY_LOOKUP, query="jwt")
    chunks = (
        _mk_preview(preview="JWT 是一种 token", chunk_id="c1"),
        _mk_preview(preview="JWT 结构", chunk_id="c2"),
    )
    coverage, missing = compute_per_subquery_coverage(
        subquery=sq, chunks=chunks, entity_hints=("jwt",)
    )
    assert missing == ()
    assert coverage == 1.0


def test_per_subquery_coverage_grades_relevance_in_mid_state() -> None:
    """召回到了但 chunk score 偏低时，coverage 必须低于 1.0 且无 missing_aspects。

    旧实现 base 恒为 1.0，只要 chunks≥2 且并非全部低于阈值就给满分；导致
    baseline 中"勉强相关"的检索（一个达标 + 一个低分）也被记为 coverage=1.0。
    新公式按 score / MIN_CHUNK_SCORE 加权，应能反映这种灰色地带。
    """

    sq = Subquery(id="sq1", intent=SUBQUERY_INTENT_PROCEDURE, query="X")
    # 一个达标 + 一个低分（不会触发 low_confidence，因为不是"all below"）
    chunks = (
        _mk_preview(score=MIN_CHUNK_SCORE, chunk_id="c1"),
        _mk_preview(score=MIN_CHUNK_SCORE / 3, chunk_id="c2"),
    )
    coverage, missing = compute_per_subquery_coverage(subquery=sq, chunks=chunks)
    # 没触发任何结构化 missing_aspects（chunks 数够、不是全低分、无 entity_hints）
    assert missing == ()
    # 但 coverage 必须落在 (0, 1) 之间，体现 chunk 相关度不齐
    assert 0.0 < coverage < 1.0


# ---------------------------------------------------------------------------
# global 规则：仅在 comparison / depends_on 场景启用
# ---------------------------------------------------------------------------


def test_global_skipped_for_independent_subqueries() -> None:
    """纯独立 definition/entity_lookup 子查询 → 不跑 global，coverage=1.0。"""

    sqs = (
        Subquery(id="sq1", intent=SUBQUERY_INTENT_ENTITY_LOOKUP, query="X"),
        Subquery(id="sq2", intent=SUBQUERY_INTENT_ENTITY_LOOKUP, query="Y"),
    )
    groups = {
        "sq1": _mk_group("sq1", (_mk_preview(doc_id="d1"),)),
        "sq2": _mk_group("sq2", (_mk_preview(doc_id="d1", chunk_id="c2"),)),
    }
    coverage, missing = compute_global_coverage(subqueries=sqs, evidence_groups=groups)
    assert coverage == 1.0
    assert missing == ()


def test_global_comparison_requires_multi_doc_source() -> None:
    """comparison_arm 两条都只命中同一个 doc_id → global_no_source_diversity。"""

    sqs = (
        Subquery(id="sq1", intent=SUBQUERY_INTENT_COMPARISON_ARM, query="A"),
        Subquery(id="sq2", intent=SUBQUERY_INTENT_COMPARISON_ARM, query="B"),
    )
    groups = {
        "sq1": _mk_group(
            "sq1",
            (
                _mk_preview(doc_id="d1", chunk_id="c1"),
                _mk_preview(doc_id="d1", chunk_id="c2"),
            ),
        ),
        "sq2": _mk_group(
            "sq2",
            (
                _mk_preview(doc_id="d1", chunk_id="c3"),
                _mk_preview(doc_id="d1", chunk_id="c4"),
            ),
        ),
    }
    coverage, missing = compute_global_coverage(subqueries=sqs, evidence_groups=groups)
    assert MISSING_GLOBAL_NO_DIVERSITY in missing
    assert coverage < 1.0
    # sanity：刚好覆盖 MIN_DOCS_MULTI 以上 doc 不应报 diversity
    assert MIN_DOCS_MULTI == 2


def test_global_target_entity_miss_emitted() -> None:
    sqs = (
        Subquery(id="sq1", intent=SUBQUERY_INTENT_COMPARISON_ARM, query="A"),
        Subquery(id="sq2", intent=SUBQUERY_INTENT_COMPARISON_ARM, query="B"),
    )
    groups = {
        "sq1": _mk_group(
            "sq1",
            (
                _mk_preview(doc_id="d1", preview="A 的介绍"),
                _mk_preview(doc_id="d1", chunk_id="c2", preview="A 概览"),
            ),
        ),
        "sq2": _mk_group(
            "sq2",
            (
                _mk_preview(doc_id="d2", chunk_id="c3", preview="另一个主题"),
                _mk_preview(doc_id="d2", chunk_id="c4", preview="无关内容"),
            ),
        ),
    }
    _coverage, missing = compute_global_coverage(
        subqueries=sqs,
        evidence_groups=groups,
        target_entities=("B",),
    )
    assert f"{MISSING_GLOBAL_ENTITY_PREFIX}B" in missing


def test_global_chain_broken_when_depends_end_has_no_chunks() -> None:
    sqs = (
        Subquery(id="sq1", intent=SUBQUERY_INTENT_ENTITY_LOOKUP, query="A"),
        Subquery(
            id="sq2",
            intent=SUBQUERY_INTENT_PROCEDURE,
            query="B",
            depends_on=("sq1",),
        ),
    )
    groups = {
        "sq1": _mk_group(
            "sq1",
            (
                _mk_preview(doc_id="d1"),
                _mk_preview(doc_id="d2", chunk_id="c2"),
            ),
        ),
        "sq2": _mk_group("sq2", ()),  # 链末空
    }
    _coverage, missing = compute_global_coverage(subqueries=sqs, evidence_groups=groups)
    assert f"{MISSING_CHAIN_BROKEN_PREFIX}sq2" in missing


# ---------------------------------------------------------------------------
# detect_gaps：端到端行为
# ---------------------------------------------------------------------------


def test_detect_gaps_all_ok() -> None:
    sqs = (Subquery(id="sq1", intent=SUBQUERY_INTENT_PROCEDURE, query="X"),)
    groups = {
        "sq1": _mk_group(
            "sq1",
            (
                _mk_preview(doc_id="d1", score=0.8),
                _mk_preview(doc_id="d2", chunk_id="c2", score=0.7),
            ),
        ),
    }
    plan = detect_gaps(subqueries=sqs, evidence_groups=groups)
    assert plan.ok is True
    assert plan.reason == ""
    assert plan.per_subquery_coverage["sq1"] == 1.0
    assert plan.global_required is False
    # per_subquery 指令仍保留一条（missing 为空，top_k_multiplier=1.0）
    assert len(plan.per_subquery) == 1
    refine = plan.per_subquery[0]
    assert refine.subquery_id == "sq1"
    assert refine.missing_aspects == ()
    assert refine.top_k_multiplier == 1.0
    assert refine.diversity_by_doc is False


def test_detect_gaps_insufficient_maps_to_top_k_bump() -> None:
    sqs = (Subquery(id="sq1", intent=SUBQUERY_INTENT_PROCEDURE, query="X"),)
    groups = {"sq1": _mk_group("sq1", (_mk_preview(score=0.8),))}
    plan = detect_gaps(subqueries=sqs, evidence_groups=groups)
    assert plan.ok is False
    assert plan.reason == "per_subquery_below_threshold"
    refine = plan.per_subquery[0]
    assert MISSING_INSUFFICIENT_CHUNKS in refine.missing_aspects
    assert refine.top_k_multiplier > 1.0
    assert refine.rerank_top_k_multiplier > 1.0


def test_detect_gaps_entity_miss_maps_to_entity_hints() -> None:
    sqs = (Subquery(id="sq1", intent=SUBQUERY_INTENT_ENTITY_LOOKUP, query="JWT"),)
    groups = {
        "sq1": _mk_group(
            "sq1",
            (
                _mk_preview(preview="totally unrelated", chunk_id="c1", score=0.8),
                _mk_preview(preview="irrelevant", chunk_id="c2", score=0.8),
            ),
        ),
    }
    plan = detect_gaps(
        subqueries=sqs,
        evidence_groups=groups,
        subquery_entity_hints={"sq1": ("JWT",)},
    )
    refine = plan.per_subquery[0]
    assert "JWT" in refine.entity_hints
    assert f"{MISSING_ENTITY_PREFIX}JWT" in refine.missing_aspects


def test_detect_gaps_global_no_diversity_sets_exclude_and_diversity_flag() -> None:
    sqs = (
        Subquery(id="sq1", intent=SUBQUERY_INTENT_COMPARISON_ARM, query="A"),
        Subquery(id="sq2", intent=SUBQUERY_INTENT_COMPARISON_ARM, query="B"),
    )
    groups = {
        "sq1": _mk_group(
            "sq1",
            (
                _mk_preview(doc_id="d1", chunk_id="c1", score=0.8),
                _mk_preview(doc_id="d1", chunk_id="c2", score=0.8),
            ),
        ),
        "sq2": _mk_group(
            "sq2",
            (
                _mk_preview(doc_id="d1", chunk_id="c3", score=0.8),
                _mk_preview(doc_id="d1", chunk_id="c4", score=0.8),
            ),
        ),
    }
    plan = detect_gaps(subqueries=sqs, evidence_groups=groups)
    assert plan.ok is False
    assert plan.global_required is True
    for refine in plan.per_subquery:
        assert refine.diversity_by_doc is True
        assert refine.per_doc_limit == 1
        assert "d1" in refine.exclude_doc_ids


def test_detect_gaps_global_entity_miss_produces_new_subquery() -> None:
    sqs = (
        Subquery(id="sq1", intent=SUBQUERY_INTENT_COMPARISON_ARM, query="A"),
        Subquery(id="sq2", intent=SUBQUERY_INTENT_COMPARISON_ARM, query="B"),
    )
    groups = {
        "sq1": _mk_group(
            "sq1",
            (
                _mk_preview(doc_id="d1", preview="A 概览", score=0.8),
                _mk_preview(doc_id="d2", chunk_id="c2", preview="A 细节", score=0.8),
            ),
        ),
        "sq2": _mk_group(
            "sq2",
            (
                _mk_preview(doc_id="d3", chunk_id="c3", preview="其它主题", score=0.8),
                _mk_preview(doc_id="d4", chunk_id="c4", preview="其它主题2", score=0.8),
            ),
        ),
    }
    plan = detect_gaps(
        subqueries=sqs,
        evidence_groups=groups,
        target_entities=("B",),
    )
    assert len(plan.new_subqueries) == 1
    new = plan.new_subqueries[0]
    assert new.intent == SUBQUERY_INTENT_ENTITY_LOOKUP
    assert new.query == "B"
    assert plan.ok is False


def test_detect_gaps_chain_broken_bumps_tail_top_k() -> None:
    sqs = (
        Subquery(id="sq1", intent=SUBQUERY_INTENT_ENTITY_LOOKUP, query="A"),
        Subquery(
            id="sq2",
            intent=SUBQUERY_INTENT_PROCEDURE,
            query="B",
            depends_on=("sq1",),
        ),
    )
    groups = {
        "sq1": _mk_group(
            "sq1",
            (
                _mk_preview(doc_id="d1", score=0.8),
                _mk_preview(doc_id="d2", chunk_id="c2", score=0.8),
            ),
        ),
        "sq2": _mk_group("sq2", ()),
    }
    plan = detect_gaps(subqueries=sqs, evidence_groups=groups)
    assert plan.ok is False
    # sq2 (链末空) 应拿到 top_k 放大的 refine
    refine_by_id = {r.subquery_id: r for r in plan.per_subquery}
    assert refine_by_id["sq2"].top_k_multiplier > 1.0


def test_detect_gaps_new_subquery_respects_max_slots() -> None:
    """已有 4 个 subquery → MAX_SUBQUERIES 用尽，不再追加新 subquery。"""

    sqs = tuple(
        Subquery(id=f"sq{i}", intent=SUBQUERY_INTENT_COMPARISON_ARM, query=f"Q{i}")
        for i in range(1, 5)
    )
    groups = {
        sq.id: _mk_group(
            sq.id,
            (_mk_preview(doc_id=f"d{sq.id}", score=0.8),) * 2,
        )
        for sq in sqs
    }
    plan = detect_gaps(
        subqueries=sqs,
        evidence_groups=groups,
        target_entities=("Z",),
    )
    # global_entity_miss 应被检测到但新 subquery 因超额被剪裁
    assert plan.new_subqueries == ()

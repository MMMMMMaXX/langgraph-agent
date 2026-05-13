"""Multi-hop Gap Detector（Phase 3 PR-2）。

设计原则与 `docs/phase3-multi-hop-rag.md` §6 对齐：

- **规则驱动**：在线链路不调 LLM judge，单次请求内开销可控。
- **两层分层**：
  * `per_subquery_coverage` 管 "这个子查询有没有拿到够用的原子证据"；
  * `global_coverage` 仅在 comparison / 含 depends_on 链的场景启用，避免
    误伤纯独立子查询（definition/entity_lookup 并列）。
- **refine 只产结构化参数**，`RefinePlan.per_subquery[*].top_k_multiplier` 等
  直接给到 `retrieve_docs_for_rag` 的新参数；**绝不拼 query 文本**（§6.3
  风险表：FTS/dense 会把 `source_diversity=true` 按字面打分）。

本模块纯函数，只依赖 `app/agents/rag/multi_hop/types.py` + 常量，不触 LangGraph。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from app.agents.rag.multi_hop.types import EvidenceGroup, EvidencePreview, Subquery
from app.constants.multi_hop import (
    MAX_SUBQUERIES,
    MIN_CHUNKS_PER_SUBQUERY,
    MIN_CHUNK_SCORE,
    MIN_DOCS_MULTI,
    SUBQUERY_INTENT_COMPARISON_ARM,
    SUBQUERY_INTENT_ENTITY_LOOKUP,
)

# ---------------------------------------------------------------------------
# Missing aspect 前缀：方便下游按 `aspect.startswith("entity_miss:")` 分派 refine。
# ---------------------------------------------------------------------------

MISSING_INSUFFICIENT_CHUNKS = "insufficient_chunks"
MISSING_LOW_CONFIDENCE = "low_confidence"
MISSING_ENTITY_PREFIX = "entity_miss:"
MISSING_GLOBAL_NO_DIVERSITY = "global_no_source_diversity"
MISSING_GLOBAL_ENTITY_PREFIX = "global_entity_miss:"
MISSING_CHAIN_BROKEN_PREFIX = "chain_broken:"

# ---------------------------------------------------------------------------
# Penalty 权重：调参点；每条 penalty 独立叠加，coverage = max(0, 1 - sum(penalty))。
#
# 注意：当前实现的 `ok` 判定只看 `missing_aspects` 是否为空（见 `detect_gaps`
# 末尾注释），**不会**把 coverage 和阈值常量做比较。这里的 penalty 数值只影响
# `RefinePlan.*_coverage` 字段（供 debug_info / 趋势分析消费），不驱动 loop
# 控制流。若未来改回阈值驱动，需要同步修改 `detect_gaps` 与本注释。
# ---------------------------------------------------------------------------

_PEN_INSUFFICIENT_CHUNKS = 0.5
_PEN_LOW_CONFIDENCE = 0.4
_PEN_ENTITY_MISS = 0.3
_PEN_GLOBAL_DIVERSITY = 0.3
_PEN_GLOBAL_ENTITY_MISS = 0.3
_PEN_CHAIN_BROKEN = 0.4

# ---------------------------------------------------------------------------
# Refine 参数映射：missing_aspect → 结构化参数。multi_hop_node 据此调用
# retrieve_docs_for_rag（§6.3）。本模块不直接接触 retrieval，纯数据。
# ---------------------------------------------------------------------------

_INSUFFICIENT_TOP_K_MULT = 1.5
_INSUFFICIENT_RERANK_MULT = 1.3
_LOW_CONFIDENCE_TOP_K_MULT = 1.5


# ===========================================================================
# 数据类型
# ===========================================================================


@dataclass(frozen=True)
class SubqueryRefine:
    """单个 subquery 的 refine 指令；由 node 翻译到 retrieval 参数。"""

    subquery_id: str
    missing_aspects: tuple[str, ...]
    top_k_multiplier: float = 1.0
    rerank_top_k_multiplier: float = 1.0
    entity_hints: tuple[str, ...] = ()
    exclude_doc_ids: frozenset[str] = field(default_factory=frozenset)
    per_doc_limit: int | None = None
    diversity_by_doc: bool = False


@dataclass(frozen=True)
class NewSubquerySuggestion:
    """`global_entity_miss` 触发时向 plan 追加的新 subquery。"""

    intent: str
    query: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class RefinePlan:
    """Gap Detector 输出。

    - `ok=True`：per-subquery 与 global coverage 均达阈值，multi_hop_node 可进入
      answer 阶段；`per_subquery` 仍会返回（空 missing_aspects）便于 debug。
    - `ok=False`：需要下一跳。`per_subquery` 中 missing_aspects 非空者需 refine；
      `new_subqueries` 若非空，multi_hop_node 应在不超 MAX_SUBQUERIES 前提下追加。
    - `global_required`：便于 node / Composer 解释"为什么没看 global"。
    """

    per_subquery: tuple[SubqueryRefine, ...]
    new_subqueries: tuple[NewSubquerySuggestion, ...]
    per_subquery_coverage: Mapping[str, float]
    global_coverage: float
    global_required: bool
    ok: bool
    reason: str = ""


# ===========================================================================
# per-subquery 规则
# ===========================================================================


def _preview_contains_entity(
    previews: tuple[EvidencePreview, ...], entity: str
) -> bool:
    """朴素包含匹配。Phase 3 MVP 不引入 NER，依赖调用方传 entity_hints。

    大小写不敏感：`JWT` / `jwt` 在 FTS/dense 侧都能命中，这里同步放宽。
    """

    if not entity:
        return False
    needle = entity.strip().lower()
    if not needle:
        return False
    return any(needle in (p.preview or "").lower() for p in previews)


def compute_per_subquery_coverage(
    *,
    subquery: Subquery,
    chunks: tuple[EvidencePreview, ...],
    entity_hints: tuple[str, ...] = (),
) -> tuple[float, tuple[str, ...]]:
    """按 §6.1 规则评估单 subquery 的 coverage。

    返回 `(coverage, missing_aspects)`。coverage 取 `max(0, 1-sum(penalty))`，
    便于 multi_hop_node 写入 `EvidenceGroup.per_subquery_coverage`。
    """

    missing: list[str] = []
    penalty = 0.0

    if len(chunks) < MIN_CHUNKS_PER_SUBQUERY:
        missing.append(MISSING_INSUFFICIENT_CHUNKS)
        penalty += _PEN_INSUFFICIENT_CHUNKS

    # chunks 非空但全部低分 → low_confidence；空 chunk 走 insufficient 上面这条。
    if chunks and all(float(c.score or 0.0) < MIN_CHUNK_SCORE for c in chunks):
        missing.append(MISSING_LOW_CONFIDENCE)
        penalty += _PEN_LOW_CONFIDENCE

    # entity_hints 若提供，则任一关键实体未出现在任何 preview 中记为 miss。
    for entity in entity_hints:
        if not _preview_contains_entity(chunks, entity):
            missing.append(f"{MISSING_ENTITY_PREFIX}{entity}")
            penalty += _PEN_ENTITY_MISS

    coverage = max(0.0, 1.0 - penalty)
    return coverage, tuple(missing)


# ===========================================================================
# global 规则（仅 comparison / depends_on 链场景启用）
# ===========================================================================


def _needs_global(subqueries: tuple[Subquery, ...]) -> bool:
    """comparison_arm 或存在 depends_on 链时才跑 global 检查。"""

    for sq in subqueries:
        if sq.intent == SUBQUERY_INTENT_COMPARISON_ARM:
            return True
        if sq.depends_on:
            return True
    return False


def _collect_covered_doc_ids(
    groups: Mapping[str, EvidenceGroup],
) -> frozenset[str]:
    doc_ids: set[str] = set()
    for group in groups.values():
        for chunk in group.chunks:
            if chunk.doc_id:
                doc_ids.add(chunk.doc_id)
    return frozenset(doc_ids)


def _all_previews(
    groups: Mapping[str, EvidenceGroup],
) -> tuple[EvidencePreview, ...]:
    collected: list[EvidencePreview] = []
    for group in groups.values():
        collected.extend(group.chunks)
    return tuple(collected)


def compute_global_coverage(
    *,
    subqueries: tuple[Subquery, ...],
    evidence_groups: Mapping[str, EvidenceGroup],
    target_entities: tuple[str, ...] = (),
) -> tuple[float, tuple[str, ...]]:
    """按 §6.2 规则评估跨 subquery 的覆盖度。

    返回 `(coverage, missing_aspects)`；若 `_needs_global` 为 False 直接回
    `(1.0, ())`，表明当前 plan 不需要 global 检查。
    """

    if not _needs_global(subqueries):
        return 1.0, ()

    missing: list[str] = []
    penalty = 0.0

    # 跨文档来源多样性：comparison / cross-doc 需要 >=2 个 doc_id 才能叙述比较。
    covered_docs = _collect_covered_doc_ids(evidence_groups)
    if len(covered_docs) < MIN_DOCS_MULTI:
        missing.append(MISSING_GLOBAL_NO_DIVERSITY)
        penalty += _PEN_GLOBAL_DIVERSITY

    # 目标实体覆盖：任一 target_entity 未在所有 preview 中出现即记 miss。
    if target_entities:
        all_previews = _all_previews(evidence_groups)
        for entity in target_entities:
            if not _preview_contains_entity(all_previews, entity):
                missing.append(f"{MISSING_GLOBAL_ENTITY_PREFIX}{entity}")
                penalty += _PEN_GLOBAL_ENTITY_MISS

    # 链路完整度：任一 depends_on 链末端 sq 无 chunk → chain_broken
    for sq in subqueries:
        if not sq.depends_on:
            continue
        group = evidence_groups.get(sq.id)
        if group is None or not group.chunks:
            missing.append(f"{MISSING_CHAIN_BROKEN_PREFIX}{sq.id}")
            penalty += _PEN_CHAIN_BROKEN

    coverage = max(0.0, 1.0 - penalty)
    return coverage, tuple(missing)


# ===========================================================================
# Missing → RefineAction 转换
# ===========================================================================


def _build_per_subquery_refine(
    subquery: Subquery,
    missing_aspects: tuple[str, ...],
) -> SubqueryRefine:
    """把 per_subquery 的 missing_aspects 翻译成结构化 refine 指令。"""

    top_k_mult = 1.0
    rerank_mult = 1.0
    entity_hints: list[str] = []

    for aspect in missing_aspects:
        if aspect == MISSING_INSUFFICIENT_CHUNKS:
            top_k_mult = max(top_k_mult, _INSUFFICIENT_TOP_K_MULT)
            rerank_mult = max(rerank_mult, _INSUFFICIENT_RERANK_MULT)
        elif aspect == MISSING_LOW_CONFIDENCE:
            top_k_mult = max(top_k_mult, _LOW_CONFIDENCE_TOP_K_MULT)
        elif aspect.startswith(MISSING_ENTITY_PREFIX):
            entity = aspect[len(MISSING_ENTITY_PREFIX) :]
            if entity:
                entity_hints.append(entity)

    return SubqueryRefine(
        subquery_id=subquery.id,
        missing_aspects=missing_aspects,
        top_k_multiplier=top_k_mult,
        rerank_top_k_multiplier=rerank_mult,
        entity_hints=tuple(entity_hints),
    )


def _apply_global_missing_to_per_sq(
    per_sq: list[SubqueryRefine],
    subqueries: tuple[Subquery, ...],
    evidence_groups: Mapping[str, EvidenceGroup],
    missing_aspects: tuple[str, ...],
) -> tuple[list[SubqueryRefine], list[NewSubquerySuggestion]]:
    """把 global missing 叠加到既有 per_sq refine 或产出新 subquery 建议。

    - `global_no_source_diversity` → 所有 per_sq 加上 `exclude_doc_ids`（已覆盖 docs）+
      `diversity_by_doc=True, per_doc_limit=1`，强制 retrieval 多样化。
    - `global_entity_miss:{e}` → 追加一个 entity_lookup 子查询，query=实体本身；
      不改动既有 per_sq（它们已经锁定各自语义）。
    - `chain_broken:{sq_id}` → 对该 sq 叠加 low_confidence 同款 refine。
    """

    covered_docs = _collect_covered_doc_ids(evidence_groups)
    needs_diversity = False
    new_subqueries: list[NewSubquerySuggestion] = []
    chain_broken_ids: set[str] = set()

    for aspect in missing_aspects:
        if aspect == MISSING_GLOBAL_NO_DIVERSITY:
            needs_diversity = True
        elif aspect.startswith(MISSING_GLOBAL_ENTITY_PREFIX):
            entity = aspect[len(MISSING_GLOBAL_ENTITY_PREFIX) :]
            if entity:
                new_subqueries.append(
                    NewSubquerySuggestion(
                        intent=SUBQUERY_INTENT_ENTITY_LOOKUP,
                        query=entity,
                    )
                )
        elif aspect.startswith(MISSING_CHAIN_BROKEN_PREFIX):
            sq_id = aspect[len(MISSING_CHAIN_BROKEN_PREFIX) :]
            if sq_id:
                chain_broken_ids.add(sq_id)

    # 剪裁新 subquery 数量，保证加回原 plan 后不超过 MAX_SUBQUERIES。
    # multi_hop_node 自己做最终拼接时仍会再校验一遍上限。
    slots_left = max(0, MAX_SUBQUERIES - len(subqueries))
    if len(new_subqueries) > slots_left:
        new_subqueries = new_subqueries[:slots_left]

    if not needs_diversity and not chain_broken_ids:
        return per_sq, new_subqueries

    rebuilt: list[SubqueryRefine] = []
    for refine in per_sq:
        top_k_mult = refine.top_k_multiplier
        rerank_mult = refine.rerank_top_k_multiplier
        exclude_docs = refine.exclude_doc_ids
        per_doc_limit = refine.per_doc_limit
        diversity = refine.diversity_by_doc

        if needs_diversity:
            exclude_docs = covered_docs
            per_doc_limit = 1
            diversity = True

        if refine.subquery_id in chain_broken_ids:
            top_k_mult = max(top_k_mult, _LOW_CONFIDENCE_TOP_K_MULT)

        rebuilt.append(
            SubqueryRefine(
                subquery_id=refine.subquery_id,
                missing_aspects=refine.missing_aspects,
                top_k_multiplier=top_k_mult,
                rerank_top_k_multiplier=rerank_mult,
                entity_hints=refine.entity_hints,
                exclude_doc_ids=exclude_docs,
                per_doc_limit=per_doc_limit,
                diversity_by_doc=diversity,
            )
        )
    return rebuilt, new_subqueries


# ===========================================================================
# 主入口
# ===========================================================================


def detect_gaps(
    *,
    subqueries: tuple[Subquery, ...],
    evidence_groups: Mapping[str, EvidenceGroup],
    subquery_entity_hints: Mapping[str, tuple[str, ...]] | None = None,
    target_entities: tuple[str, ...] = (),
) -> RefinePlan:
    """一次性跑完 per-subquery + global 判定，返回 RefinePlan。

    multi_hop_node 调用约定：
    1. 先按 subquery 跑 retrieval，组装 `evidence_groups`（`EvidenceGroup.chunks`
       只填 preview，不填全文）；
    2. 调 `detect_gaps(...)`，若 `plan.ok` 即退出 loop；
    3. 否则按 `plan.per_subquery` + `plan.new_subqueries` 触发下一跳 retrieval。

    `subquery_entity_hints` 为 **可选**：调用方若从 decomposer 提取了每 sq 的关键
    实体，可在此传入；未提供时只跑 chunk 数 / score 类规则，不做 entity miss 惩罚。
    """

    hints_by_sq = subquery_entity_hints or {}
    per_sq_refines: list[SubqueryRefine] = []
    coverages: dict[str, float] = {}

    for sq in subqueries:
        group = evidence_groups.get(sq.id)
        chunks = group.chunks if group is not None else ()
        coverage, missing = compute_per_subquery_coverage(
            subquery=sq,
            chunks=chunks,
            entity_hints=hints_by_sq.get(sq.id, ()),
        )
        coverages[sq.id] = coverage
        per_sq_refines.append(_build_per_subquery_refine(sq, missing))

    global_required = _needs_global(subqueries)
    global_coverage, global_missing = compute_global_coverage(
        subqueries=subqueries,
        evidence_groups=evidence_groups,
        target_entities=target_entities,
    )

    per_sq_refines, new_subqueries = _apply_global_missing_to_per_sq(
        per_sq_refines,
        subqueries,
        evidence_groups,
        global_missing,
    )

    # `ok` 判定以 **missing_aspects 是否全空** 为唯一真源：只要检测到任何
    # gap（哪怕 coverage 数值恰好等于阈值），都应走下一跳 refine。阈值和 coverage
    # 分数只保留给 debug_info / 趋势分析用，避免"阈值边界值"漏报。
    per_sq_has_missing = any(r.missing_aspects for r in per_sq_refines)
    global_has_missing = bool(global_missing)
    ok = not per_sq_has_missing and not global_has_missing and not new_subqueries

    reason = ""
    if per_sq_has_missing:
        reason = "per_subquery_below_threshold"
    elif global_has_missing and not new_subqueries:
        reason = "global_below_threshold"
    elif new_subqueries:
        reason = "global_entity_miss"

    return RefinePlan(
        per_subquery=tuple(per_sq_refines),
        new_subqueries=tuple(new_subqueries),
        per_subquery_coverage=coverages,
        global_coverage=global_coverage,
        global_required=global_required,
        ok=ok,
        reason=reason,
    )


__all__ = [
    "MISSING_CHAIN_BROKEN_PREFIX",
    "MISSING_ENTITY_PREFIX",
    "MISSING_GLOBAL_ENTITY_PREFIX",
    "MISSING_GLOBAL_NO_DIVERSITY",
    "MISSING_INSUFFICIENT_CHUNKS",
    "MISSING_LOW_CONFIDENCE",
    "NewSubquerySuggestion",
    "RefinePlan",
    "SubqueryRefine",
    "compute_global_coverage",
    "compute_per_subquery_coverage",
    "detect_gaps",
]

"""Phase 3 PR-3：Multi-hop RAG 主节点。

职责：
1. 读取 base_query（优先 rewritten_query，兜底 latest user message），
   走 decomposer 切分子问题。
2. 对每个子问题调 `retrieve_docs_for_rag` 做一次检索；把结果汇总成：
   - `evidence_groups`（preview only，写进 debug_info）
   - `full_chunks`(局部 dict[chunk_id, doc_hit]，仅节点内存在，不进 State)
3. 调 `detect_gaps(...)` 评估覆盖度；`plan.ok=False` 且 hop<MAX_HOPS 时按
   `RefinePlan.per_subquery` / `new_subqueries` 发起下一跳 retrieval，最多 MAX_HOPS
   轮；hop 耗尽仍有 gap 标 `budget_exceeded`（step=PARTIAL，有证据时不算整体失败）。
4. 从 full_chunks 里按 hybrid score 挑前 `MAX_TOTAL_CHUNKS` 条，走
   `answer_with_doc_hits` 生成最终 answer（策略固定为 `ANSWER_STRATEGY_MULTI_HOP`）。
5. 合成 §4.1 的 pseudo-step 写入 `step_results["mh1"]`；同步把最终 answer 写到
   `agent_outputs[ROUTE_MULTI_HOP_AGENT]`，再交给 Verifier / Composer 做覆盖校验
   和直通合成。

降级路径（不双写 agent_outputs）：
- decompose 失败 → `run_single_hop_retrieval_answer(base_query)` + degrade_reason
- 所有 subquery 都 0 chunk → step.status=failed + evidence_empty
- refine loop 用完 hop budget 仍有 gap → step.status=PARTIAL + budget_exceeded
- answer LLM 异常 → step.status=failed + answer_llm_failed

硬约束（与 PR-3 plan 同步）：
- State 顶层**不**出现 evidence_groups / 全文字段；全量 chunk 只在局部变量。
- `agent_outputs` 只写 `ROUTE_MULTI_HOP_AGENT` 一个 key。
- 本节点不触发 rag_agent_node，fallback 走纯函数 `run_single_hop_retrieval_answer`。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.agents.rag.answer_flow import (
    answer_with_doc_hits,
    build_multi_hop_answer_strategy,
    run_single_hop_retrieval_answer,
)
from app.agents.rag.citations import get_chunk_identifier
from app.agents.rag.doc_pipeline import retrieve_docs_for_rag
from app.agents.rag.multi_hop.decompose import decompose_query
from app.agents.rag.multi_hop.gap import (
    RefinePlan,
    SubqueryRefine,
    detect_gaps,
)
from app.agents.rag.multi_hop.types import (
    DecomposeResult,
    EvidenceGroup,
    EvidencePreview,
    Subquery,
)
from app.constants.multi_hop import (
    DEGRADE_REASON_ANSWER_LLM_FAILED,
    DEGRADE_REASON_BUDGET_EXCEEDED,
    DEGRADE_REASON_EVIDENCE_EMPTY,
    EVIDENCE_PREVIEW_MAX_CHARS,
    MAX_HOPS,
    MAX_SUBQUERIES,
    MAX_TOTAL_CHUNKS,
    MIN_CHUNK_SCORE,
    MIN_CHUNKS_PER_SUBQUERY,
    MULTI_HOP_DEBUG_KEY,
    MULTI_HOP_STEP_ID,
)
from app.constants.routes import ROUTE_MULTI_HOP_AGENT
from app.constants.workflow import (
    STEP_AGENT_RAG,
    STEP_STATUS_FAILED,
    STEP_STATUS_SUCCEEDED,
    TASK_TYPE_MULTI_HOP_RAG,
    WORKFLOW_STATUS_FAILED,
    WORKFLOW_STATUS_PARTIAL,
    WORKFLOW_STATUS_SUCCEEDED,
)
from app.state import AgentState
from app.streaming import build_answer_streamer
from app.utils.errors import build_error_info
from app.utils.logger import log_node, now_ms, preview

# `multi_hop_plan.subqueries[*]` 只登记这几个 key，避免 State 体积膨胀与全文泄漏。
_PLAN_SUBQUERY_KEEP_KEYS = ("id", "intent", "query")

# Multi-hop pseudo-step 固定 id 在 `app/constants/multi_hop.py:MULTI_HOP_STEP_ID`
# 统一维护，Composer / Verifier / eval 共享同一常量，避免字面量副本。

# debug_info 下的多跳分层 key 已抽到 `app/constants/multi_hop.py:MULTI_HOP_DEBUG_KEY`，
# Composer / Verifier / eval / 前端共享，避免字面量"multi_hop"散落各处。
_BASE_QUERY_SOURCE_REWRITTEN = "rewritten"
_BASE_QUERY_SOURCE_LATEST_MESSAGE = "latest_user_message"


def _extract_base_query(state: AgentState) -> tuple[str, str]:
    """返回 `(base_query, source)`。

    优先用 rewritten_query；为空 / 纯空白时兜底最新 user message。
    两者都取不到时返回 `("", "")`，由调用方走 evidence_empty 失败路径。
    """

    rewritten = (state.get("rewritten_query") or "").strip()
    if rewritten:
        return rewritten, _BASE_QUERY_SOURCE_REWRITTEN

    messages = state.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            if content:
                return content, _BASE_QUERY_SOURCE_LATEST_MESSAGE
    return "", ""


def _truncate_preview(text: str) -> str:
    """统一的 preview 截断；`EvidencePreview.preview` 必须 ≤ 常量上限。"""

    if not text:
        return ""
    return text[:EVIDENCE_PREVIEW_MAX_CHARS]


def _doc_hit_to_preview(doc_hit: dict, ref_index: int) -> EvidencePreview:
    return EvidencePreview(
        doc_id=str(doc_hit.get("doc_id", "")),
        chunk_id=get_chunk_identifier(doc_hit),
        ref=f"[{ref_index}]",
        score=round(float(doc_hit.get("score", 0.0) or 0.0), 4),
        preview=_truncate_preview(doc_hit.get("content", "")),
    )


def _retrieve_for_subquery(
    subquery: Subquery,
    refine: SubqueryRefine | None = None,
) -> tuple[list[dict], dict, list[str], dict[str, float]]:
    """单 subquery 检索：返回 merged_doc_hits + retrieval debug。

    当 `refine` 非 None 时透传 PR-2 的结构化参数（top_k 倍增、exclude_doc_ids、
    per_doc_limit、entity_hints、source_diversity），对应 gap detector 的
    missing_aspects → refine 指令映射。不传则走默认值，等价于首跳行为。
    """

    kwargs: dict = {}
    if refine is not None:
        kwargs.update(
            top_k_multiplier=refine.top_k_multiplier,
            rerank_top_k_multiplier=refine.rerank_top_k_multiplier,
            entity_hints=refine.entity_hints,
            per_doc_limit=refine.per_doc_limit,
            force_source_diversity=refine.diversity_by_doc,
        )
        if refine.exclude_doc_ids:
            kwargs["exclude_doc_ids"] = refine.exclude_doc_ids

    result = retrieve_docs_for_rag(subquery.query, **kwargs)
    return (
        result.merged_doc_hits,
        result.retrieval_debug,
        result.errors,
        result.timings_ms,
    )


def _coverage_from_scores(scores: list[float]) -> float:
    """共享的 chunk 级相关度加权 coverage 公式（0~1）。

    ``scores`` 既可能来自原始 hit dict（首跳），也可能来自 EvidencePreview
    （refine 合并阶段），统一抽出 score 列表后走同一公式，避免两个调用路径
    各自写一份。
    """

    if not scores:
        return 0.0
    if MIN_CHUNK_SCORE <= 0 or MIN_CHUNKS_PER_SUBQUERY <= 0:
        return 1.0  # 退化为旧二元行为，避免除 0

    top = sorted(scores, reverse=True)[:MIN_CHUNKS_PER_SUBQUERY]
    contribs = [min(1.0, max(0.0, s) / MIN_CHUNK_SCORE) for s in top]
    coverage = sum(contribs) / MIN_CHUNKS_PER_SUBQUERY
    return round(min(1.0, coverage), 4)


def _compute_per_subquery_coverage(hits: list[dict]) -> float:
    """按 chunk 相关度 + 数量加权计算 per_subquery_coverage（0~1）。

    旧实现是 `1.0 if hits else 0.0` 的二元开关——只要有任意 hit 就给满分，导致
    coverage 系统性虚高（baseline 中常见 avg_global_coverage=1.0 但 LLM 仍判
    "资料不足"），refine_loop 也无法据此识别"召回了但相关性差"的退化情况。

    新实现按 top-N hit 的 score 相对于 `MIN_CHUNK_SCORE` 的归一化贡献做平均：
      per_chunk_contrib = min(1.0, score / MIN_CHUNK_SCORE)
      coverage = sum(top N contribs) / MIN_CHUNKS_PER_SUBQUERY
    其中 N = MIN_CHUNKS_PER_SUBQUERY。这样：
      - 0 hits → 0
      - 1 个 score≥阈值的 hit → 0.5（仍未达 PER_SUBQUERY_OK_THRESHOLD=0.6，会触发 refine）
      - 2 个 score≥阈值的 hit → 1.0
      - N 个低分 hit → 按比例衰减，反映相关性差
    既保留量上达标即满分的稳定性，也让 refine_loop 能感知到弱召回。
    """

    return _coverage_from_scores([float(hit.get("score", 0.0) or 0.0) for hit in hits])


def _build_evidence_group(
    subquery: Subquery,
    hits: list[dict],
    hop: int,
    ref_start: int,
) -> tuple[EvidenceGroup, int]:
    previews = tuple(
        _doc_hit_to_preview(hit, ref_start + i) for i, hit in enumerate(hits)
    )
    return (
        EvidenceGroup(
            subquery_id=subquery.id,
            chunks=previews,
            per_subquery_coverage=_compute_per_subquery_coverage(hits),
            missing_aspects=(),
            hop=hop,
        ),
        ref_start + len(previews),
    )


def _select_top_chunks(full_pool: list[dict], limit: int) -> list[dict]:
    """按 hybrid score 降序挑前 N；相同 chunk_id 去重。"""

    seen: set[str] = set()
    ordered: list[dict] = []
    for hit in sorted(
        full_pool,
        key=lambda d: float(d.get("score", 0.0) or 0.0),
        reverse=True,
    ):
        cid = get_chunk_identifier(hit)
        if cid in seen:
            continue
        seen.add(cid)
        ordered.append(hit)
        if len(ordered) >= limit:
            break
    return ordered


def _build_plan_summary(decompose: DecomposeResult) -> dict:
    """State.multi_hop_plan 只放元数据，不放全量 RefinePlan / chunk。"""

    return {
        "task_type": TASK_TYPE_MULTI_HOP_RAG,
        "subqueries": [
            {k: getattr(sq, k) for k in _PLAN_SUBQUERY_KEEP_KEYS}
            for sq in decompose.subqueries
        ],
        "degraded_to_single_hop": decompose.degraded_to_single_hop,
    }


def _build_chunk_id_to_sq_ids(
    evidence_groups: dict[str, EvidenceGroup],
) -> dict[str, tuple[str, ...]]:
    """chunk_id → 出现过的 sq_id 元组（稳定有序）。

    Composer 的 `_build_citations_from_mh` 会读 citation.subquery_ids 做展示；
    refine loop 可能让同一 chunk 被多个 sq 引用，所以这里保留全部关联 sq。
    """

    mapping: dict[str, list[str]] = {}
    for sq_id, group in evidence_groups.items():
        for chunk in group.chunks:
            chunk_id = chunk.chunk_id
            if not chunk_id:
                continue
            bucket = mapping.setdefault(chunk_id, [])
            if sq_id not in bucket:
                bucket.append(sq_id)
    return {cid: tuple(ids) for cid, ids in mapping.items()}


def _annotate_citations_with_subquery_ids(
    citations: list[dict],
    chunk_id_to_sq_ids: dict[str, tuple[str, ...]],
) -> list[dict]:
    """给 citation 补 `subquery_ids` 字段（空时写 ())，保持字段齐整。

    Composer/前端据此展示 "[1] 资料来自 sq1, sq3"；非多跳链路没有该字段，
    PR-4 的 `_build_citations_from_mh` 会按 `(doc_id, ref)` 合并。
    """

    enriched: list[dict] = []
    for citation in citations:
        chunk_id = citation.get("chunk_id", "")
        sq_ids = chunk_id_to_sq_ids.get(chunk_id, ())
        enriched.append({**citation, "subquery_ids": sq_ids})
    return enriched


def _evidence_group_to_dict(group: EvidenceGroup) -> dict:
    data = asdict(group)
    # preview 已在入口截断；这里再保险一次，避免后续改动漏 check。
    data["chunks"] = [
        {**chunk, "preview": _truncate_preview(chunk.get("preview", ""))}
        for chunk in data["chunks"]
    ]
    return data


def _pseudo_step(
    *,
    status: str,
    output: str,
    citations: list[dict],
    hop_count: int,
    degrade_reason: str = "",
    extra: dict | None = None,
) -> dict:
    """合成 `step_results["mh1"]`。

    字段形状对齐 §4.1：后续 Composer 直通逻辑会读 status / output / citations /
    meta.hop_count。本 PR 只负责写这些字段，Composer 适配在 PR-4 做。
    """

    meta = {"hop_count": hop_count}
    if degrade_reason:
        meta["degrade_reason"] = degrade_reason
    if extra:
        meta.update(extra)
    return {
        "id": MULTI_HOP_STEP_ID,
        "status": status,
        "output": output,
        "citations": citations,
        "meta": meta,
    }


def _empty_pseudo_step_for_failure(
    *,
    reason: str,
    hop_count: int,
) -> dict:
    return _pseudo_step(
        status=STEP_STATUS_FAILED,
        output="",
        citations=[],
        hop_count=hop_count,
        degrade_reason=reason,
    )


def _run_retrieval_round(
    *,
    subquery: Subquery,
    hop: int,
    refine: SubqueryRefine | None,
    existing_group: EvidenceGroup | None,
    full_chunks_pool: list[dict],
    seen_chunk_ids: set[str],
    ref_counter: int,
    debug: dict,
    errors: list[str],
) -> tuple[EvidenceGroup, int]:
    """一次 sq 检索 + 副作用更新：pool / preview / debug。

    - `existing_group` 非空时表示 refine 再次检索：把新 hits 和旧 previews 合并去重，
      重建 EvidenceGroup（ref 序号在全局 ref_counter 中连续分配）。
    - 副作用只落在传入的 mutable 容器上，调用点负责把容器挂到 state。
    """

    started = now_ms()
    merged, retrieval_debug, sq_errors, _sq_timings = _retrieve_for_subquery(
        subquery, refine=refine
    )
    elapsed = round(now_ms() - started, 2)
    # retrieval_ms_per_subquery：累加而非覆盖，便于看到多次 refine 成本
    prev_ms = debug["retrieval_ms_per_subquery"].get(subquery.id, 0.0)
    debug["retrieval_ms_per_subquery"][subquery.id] = round(prev_ms + elapsed, 2)
    errors.extend(sq_errors)

    debug_entry = {
        "hit_count": len(merged),
        "refine_docs_dropped": retrieval_debug.get("refine_docs_dropped", 0),
        "hop": hop,
    }
    debug.setdefault("per_subquery_retrieval_debug", {}).setdefault(
        subquery.id, []
    ).append(debug_entry)

    # merge hits into full_chunks_pool
    for hit in merged:
        cid = get_chunk_identifier(hit)
        if not cid or cid in seen_chunk_ids:
            continue
        seen_chunk_ids.add(cid)
        full_chunks_pool.append(hit)
        if len(full_chunks_pool) >= MAX_TOTAL_CHUNKS * 2:
            # 软上限；真正 MAX_TOTAL_CHUNKS 截断在 _select_top_chunks 做。
            break

    # 为这次 sq 产出/更新 EvidenceGroup：把新 hits 的 preview 合入旧 preview
    # （按 chunk_id 去重），以便 detect_gaps 看到累计证据。
    new_group, ref_counter = _build_evidence_group(subquery, merged, hop, ref_counter)
    if existing_group is not None:
        seen_preview_cids: set[str] = set()
        combined: list[EvidencePreview] = []
        for preview_item in (*existing_group.chunks, *new_group.chunks):
            if preview_item.chunk_id in seen_preview_cids:
                continue
            seen_preview_cids.add(preview_item.chunk_id)
            combined.append(preview_item)
        new_group = EvidenceGroup(
            subquery_id=subquery.id,
            chunks=tuple(combined),
            per_subquery_coverage=_coverage_from_scores(
                [float(p.score or 0.0) for p in combined]
            ),
            missing_aspects=(),
            hop=hop,
        )

    debug["per_subquery_coverage"][subquery.id] = new_group.per_subquery_coverage
    return new_group, ref_counter


def _refine_has_action(refine: SubqueryRefine) -> bool:
    """判断 SubqueryRefine 是否真的需要再跑一次检索。

    只看 missing_aspects 是不够的：`global_no_source_diversity` 命中时
    gap detector 会往 per_subquery 的 `exclude_doc_ids / per_doc_limit /
    diversity_by_doc` 写值（见 gap._apply_global_missing_to_per_sq），但不改
    missing_aspects。若按 `missing_aspects` 判空会跳过这次 refine，
    "两个 arm 都命中同一个 doc" 的场景就会空转到 budget_exceeded。
    """

    if refine.missing_aspects:
        return True
    if refine.top_k_multiplier > 1.0 or refine.rerank_top_k_multiplier > 1.0:
        return True
    if refine.entity_hints or refine.exclude_doc_ids:
        return True
    if refine.per_doc_limit is not None or refine.diversity_by_doc:
        return True
    return False


def _suggestion_to_subquery(
    suggestion,  # NewSubquerySuggestion
    existing_ids: set[str],
) -> Subquery:
    """把 gap detector 的 new_subquery 建议升级为正式 Subquery。

    id 以 `sq{N}` 线性递增，避免与 decomposer 产出撞号。调用方须保证不超过
    MAX_SUBQUERIES（gap detector 已做剪裁，此处再校验一次防御性兜底）。
    """

    idx = len(existing_ids) + 1
    new_id = f"sq{idx}"
    while new_id in existing_ids:
        idx += 1
        new_id = f"sq{idx}"
    return Subquery(
        id=new_id,
        intent=suggestion.intent,
        query=suggestion.query,
        depends_on=suggestion.depends_on,
    )


def multi_hop_node(state: AgentState) -> AgentState:
    """Multi-hop RAG 主节点；见模块 docstring。"""

    started_at_ms = now_ms()
    base_query, base_query_source = _extract_base_query(state)
    debug: dict = {
        "base_query_source": base_query_source,
        "base_query_preview": preview(base_query, 120),
        "retrieval_ms_per_subquery": {},
        "per_subquery_coverage": {},
        "hop_count": 0,
        "degrade_reason": "",
        "evidence_groups_preview": [],
    }
    errors: list[str] = []

    # --- 空 query：直接失败，不做检索 ---
    if not base_query:
        step = _empty_pseudo_step_for_failure(
            reason=DEGRADE_REASON_EVIDENCE_EMPTY,
            hop_count=0,
        )
        debug["degrade_reason"] = DEGRADE_REASON_EVIDENCE_EMPTY
        return _finalize(
            state=state,
            answer="",
            step=step,
            hop_count=0,
            multi_hop_plan={"task_type": TASK_TYPE_MULTI_HOP_RAG, "subqueries": []},
            debug=debug,
            errors=errors,
            workflow_status=WORKFLOW_STATUS_FAILED,
            total_ms=now_ms() - started_at_ms,
        )

    # streamer 提前构造：fallback / multi-hop 主路径都要透传 on_delta，保持
    # 流式行为对外一致（否则 fallback 会静默丢失流式能力）。
    on_delta, stream_state = build_answer_streamer(state, ROUTE_MULTI_HOP_AGENT)

    # --- 1. decompose ---
    decompose_started = now_ms()
    role = getattr(state.get("auth"), "role", "user") or "user"
    decompose_result = decompose_query(rewritten_query=base_query, role=role)
    debug["decompose_ms"] = round(now_ms() - decompose_started, 2)
    debug["decompose_reason"] = decompose_result.reason
    debug["decompose_error_code"] = decompose_result.error_code

    multi_hop_plan = _build_plan_summary(decompose_result)

    # --- 2. decompose 失败 → 纯函数单跳 fallback ---
    if decompose_result.degraded_to_single_hop:
        degrade_reason = decompose_result.error_code or DEGRADE_REASON_EVIDENCE_EMPTY
        debug["degrade_reason"] = degrade_reason
        try:
            fallback = run_single_hop_retrieval_answer(
                base_query=base_query,
                on_delta=on_delta,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                build_error_info(exc, stage="multi_hop_fallback", source="rag")
            )
            step = _empty_pseudo_step_for_failure(
                reason=DEGRADE_REASON_ANSWER_LLM_FAILED,
                hop_count=1,
            )
            debug["degrade_reason"] = DEGRADE_REASON_ANSWER_LLM_FAILED
            return _finalize(
                state=state,
                answer="",
                step=step,
                hop_count=1,
                multi_hop_plan=multi_hop_plan,
                debug=debug,
                errors=errors,
                workflow_status=WORKFLOW_STATUS_FAILED,
                total_ms=now_ms() - started_at_ms,
            )

        errors.extend(fallback.errors)
        debug["answer_strategy"] = fallback.answer_strategy.get("name", "")
        debug["single_hop_fallback"] = True
        debug["single_hop_timings_ms"] = fallback.timings_ms
        has_evidence = bool(fallback.merged_doc_hits)
        status = STEP_STATUS_SUCCEEDED if fallback.answer else STEP_STATUS_FAILED
        workflow_status = (
            WORKFLOW_STATUS_PARTIAL if has_evidence else WORKFLOW_STATUS_FAILED
        )
        if status == STEP_STATUS_FAILED:
            workflow_status = WORKFLOW_STATUS_FAILED
        step = _pseudo_step(
            status=status,
            output=fallback.answer,
            citations=fallback.citations,
            hop_count=1,
            degrade_reason=degrade_reason,
            extra={"single_hop_fallback": True},
        )
        result_state = _finalize(
            state=state,
            answer=fallback.answer,
            step=step,
            hop_count=1,
            multi_hop_plan=multi_hop_plan,
            debug=debug,
            errors=errors,
            workflow_status=workflow_status,
            total_ms=now_ms() - started_at_ms,
        )
        if stream_state["used"]:
            result_state["streamed_answer"] = True
        return result_state

    # --- 3. 多跳 retrieve + refine loop ---
    # 首跳：所有 sq 各跑一次 retrieval；后续 hop 按 gap detector 的 RefinePlan 扩展：
    # - per_subquery[*].missing_aspects 非空 → 带 refine 参数二次检索同一 sq
    # - new_subqueries → 追加新 sq（不超 MAX_SUBQUERIES）并做首次检索
    # hop 用尽仍有 gap：budget_exceeded + PARTIAL（有证据时不算整体 FAIL）。
    subqueries: list[Subquery] = list(decompose_result.subqueries)
    full_chunks_pool: list[dict] = []
    seen_chunk_ids: set[str] = set()
    evidence_groups: dict[str, EvidenceGroup] = {}
    ref_counter = 1
    hop = 1

    # 首跳
    for sq in subqueries:
        group, ref_counter = _run_retrieval_round(
            subquery=sq,
            hop=hop,
            refine=None,
            existing_group=None,
            full_chunks_pool=full_chunks_pool,
            seen_chunk_ids=seen_chunk_ids,
            ref_counter=ref_counter,
            debug=debug,
            errors=errors,
        )
        evidence_groups[sq.id] = group

    # refine loop
    refine_plan: RefinePlan | None = detect_gaps(
        subqueries=tuple(subqueries),
        evidence_groups=evidence_groups,
    )
    last_refine_plan = refine_plan

    while refine_plan is not None and not refine_plan.ok and hop < MAX_HOPS:
        hop += 1

        # 先追加 new_subqueries 并首跳 retrieve（respect MAX_SUBQUERIES）
        existing_ids = {sq.id for sq in subqueries}
        for suggestion in refine_plan.new_subqueries:
            if len(subqueries) >= MAX_SUBQUERIES:
                break
            new_sq = _suggestion_to_subquery(suggestion, existing_ids)
            subqueries.append(new_sq)
            existing_ids.add(new_sq.id)
            group, ref_counter = _run_retrieval_round(
                subquery=new_sq,
                hop=hop,
                refine=None,
                existing_group=None,
                full_chunks_pool=full_chunks_pool,
                seen_chunk_ids=seen_chunk_ids,
                ref_counter=ref_counter,
                debug=debug,
                errors=errors,
            )
            evidence_groups[new_sq.id] = group

        # 再对既有 sq 带 refine 参数二次检索
        sq_by_id = {sq.id: sq for sq in subqueries}
        for sub_refine in refine_plan.per_subquery:
            # 不能只看 missing_aspects：global source_diversity 只写结构化参数
            # 不改 missing_aspects（见 _refine_has_action 注释）。
            if not _refine_has_action(sub_refine):
                continue
            target = sq_by_id.get(sub_refine.subquery_id)
            if target is None:
                continue
            group, ref_counter = _run_retrieval_round(
                subquery=target,
                hop=hop,
                refine=sub_refine,
                existing_group=evidence_groups.get(target.id),
                full_chunks_pool=full_chunks_pool,
                seen_chunk_ids=seen_chunk_ids,
                ref_counter=ref_counter,
                debug=debug,
                errors=errors,
            )
            evidence_groups[target.id] = group

        refine_plan = detect_gaps(
            subqueries=tuple(subqueries),
            evidence_groups=evidence_groups,
        )
        last_refine_plan = refine_plan

    # decompose 后的 plan 可能追加了 new_subqueries；把最终 plan summary 同步更新。
    # 只保留 id/intent/query（与初始 summary 结构一致）。
    multi_hop_plan["subqueries"] = [
        {k: getattr(sq, k) for k in _PLAN_SUBQUERY_KEEP_KEYS} for sq in subqueries
    ]

    debug["hop_count"] = hop
    debug["evidence_groups_preview"] = [
        _evidence_group_to_dict(g) for g in evidence_groups.values()
    ]
    debug["refine_loop"] = {
        "final_plan_ok": bool(last_refine_plan and last_refine_plan.ok),
        "final_plan_reason": (last_refine_plan.reason if last_refine_plan else ""),
        "global_coverage": (
            last_refine_plan.global_coverage if last_refine_plan else 1.0
        ),
    }

    # --- 4. 证据为空 → 失败 ---
    if not full_chunks_pool:
        debug["degrade_reason"] = DEGRADE_REASON_EVIDENCE_EMPTY
        step = _empty_pseudo_step_for_failure(
            reason=DEGRADE_REASON_EVIDENCE_EMPTY,
            hop_count=hop,
        )
        return _finalize(
            state=state,
            answer="",
            step=step,
            hop_count=hop,
            multi_hop_plan=multi_hop_plan,
            debug=debug,
            errors=errors,
            workflow_status=WORKFLOW_STATUS_FAILED,
            total_ms=now_ms() - started_at_ms,
        )

    # --- 5. 汇总 chunks 生成最终 answer ---
    top_chunks = _select_top_chunks(full_chunks_pool, MAX_TOTAL_CHUNKS)

    try:
        answer, citations, strategy, _compression, ans_errors, ans_ms = (
            answer_with_doc_hits(
                question=base_query,
                doc_hits=top_chunks,
                query_type="",
                strategy=build_multi_hop_answer_strategy(),
                on_delta=on_delta,
            )
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(build_error_info(exc, stage="multi_hop_answer", source="llm"))
        debug["degrade_reason"] = DEGRADE_REASON_ANSWER_LLM_FAILED
        step = _empty_pseudo_step_for_failure(
            reason=DEGRADE_REASON_ANSWER_LLM_FAILED,
            hop_count=hop,
        )
        return _finalize(
            state=state,
            answer="",
            step=step,
            hop_count=hop,
            multi_hop_plan=multi_hop_plan,
            debug=debug,
            errors=errors,
            workflow_status=WORKFLOW_STATUS_FAILED,
            total_ms=now_ms() - started_at_ms,
        )

    errors.extend(ans_errors)
    debug["answer_strategy"] = strategy.get("name", "")
    debug["answer_ms"] = ans_ms
    debug["selected_chunk_count"] = len(top_chunks)

    # Phase 3 PR-4 契约：给每条 citation 打上 subquery_ids（保留 0..N 关联），
    # 让 Composer 的 `_build_citations_from_mh` 可以按 (doc_id, ref) 合并展示。
    citations = _annotate_citations_with_subquery_ids(
        citations,
        _build_chunk_id_to_sq_ids(evidence_groups),
    )

    # Phase 3 PR-4：把"哪些 sq 没召回任何 chunk"落到 step.meta，供 Verifier
    # 直接消费（RISK_WARN_MULTI_HOP_COVERAGE）。不走 debug_info——debug 只做展示，
    # 跨节点语义信号应该留在结构化状态里。
    missing_coverage_sq_ids: list[str] = [
        sq.id
        for sq in subqueries
        if not (evidence_groups.get(sq.id) and evidence_groups[sq.id].chunks)
    ]

    # 状态判定：gap 未闭合 → PARTIAL + budget_exceeded（有答案但覆盖不完整，
    # 由 Composer/前端据此提示用户"可能有子问题未回答"）；全部 sq 覆盖 ok →
    # SUCCEEDED。只要进到这里 full_chunks_pool 非空，就一定有证据可用。
    plan_ok = bool(last_refine_plan and last_refine_plan.ok)
    mh_extra: dict = {}
    if missing_coverage_sq_ids:
        mh_extra["missing_coverage_sq_ids"] = missing_coverage_sq_ids
    if plan_ok:
        step = _pseudo_step(
            status=STEP_STATUS_SUCCEEDED,
            output=answer,
            citations=citations,
            hop_count=hop,
            extra=mh_extra or None,
        )
        workflow_status = WORKFLOW_STATUS_SUCCEEDED
    else:
        debug["degrade_reason"] = DEGRADE_REASON_BUDGET_EXCEEDED
        mh_extra["final_plan_reason"] = (
            last_refine_plan.reason if last_refine_plan else ""
        )
        step = _pseudo_step(
            status=STEP_STATUS_SUCCEEDED if answer else STEP_STATUS_FAILED,
            output=answer,
            citations=citations,
            hop_count=hop,
            degrade_reason=DEGRADE_REASON_BUDGET_EXCEEDED,
            extra=mh_extra,
        )
        workflow_status = WORKFLOW_STATUS_PARTIAL if answer else WORKFLOW_STATUS_FAILED

    result_state = _finalize(
        state=state,
        answer=answer,
        step=step,
        hop_count=hop,
        multi_hop_plan=multi_hop_plan,
        debug=debug,
        errors=errors,
        workflow_status=workflow_status,
        total_ms=now_ms() - started_at_ms,
    )
    if stream_state["used"]:
        result_state["streamed_answer"] = True
    return result_state


def _finalize(
    *,
    state: AgentState,
    answer: str,
    step: dict,
    hop_count: int,
    multi_hop_plan: dict,
    debug: dict,
    errors: list[str],
    workflow_status: str,
    total_ms: float,
) -> AgentState:
    """统一的 State 写入点：保证只在一处组装返回值。"""

    debug_payload = dict(debug)
    debug_payload["total_ms"] = round(total_ms, 2)
    # 把 refine_loop.global_coverage 同步到 debug 顶层。
    # 历史上 coverage 只写到 `refine_loop.global_coverage`，eval / 前端 / 监控
    # 想读 mh_debug["global_coverage"] 时全部拿到 0.0，导致聚合层把覆盖率误报为零。
    # 在 _finalize 统一镜像，所有进入此处的分支都会得到一致的顶层字段。
    if "global_coverage" not in debug_payload:
        refine_loop = debug_payload.get("refine_loop") or {}
        if "global_coverage" in refine_loop:
            debug_payload["global_coverage"] = refine_loop["global_coverage"]
    # 把最终交付给 Composer 的 citations 也带进 debug，让 eval / trace 可以
    # 直接看到 multi-hop 真正引用的 doc/chunk，而不必绕道 step_results
    # （后者不会出现在 API debug 响应里）。
    debug_payload["citations"] = list(step.get("citations") or [])
    if errors:
        debug_payload["errors"] = errors

    # 统一 State 契约：Composer / Verifier 只读 `state["plan"]`。
    # multi-hop 没有真实 plan，这里合成一条 pseudo plan，只包含 mh1 一条 step：
    # - 让 Composer 的 `task_type == multi_hop_rag` 直通分支命中；
    # - 让 Verifier 的 `not steps` early-return 不再吞掉 multi-hop 路径，
    #   使 `_check_multi_hop_coverage` 可以执行。
    # `multi_hop_plan`（去字面化的 subqueries 摘要）仍单独透传给 debug / eval。
    pseudo_plan: dict[str, Any] = {
        "task_type": TASK_TYPE_MULTI_HOP_RAG,
        "steps": [
            {
                "id": step["id"],
                "agent": STEP_AGENT_RAG,
                "purpose": "multi_hop_rag",
                "tool": None,
                "args": {},
                "query": None,
                "depends_on": [],
            }
        ],
        "compose_goal": "",
    }

    next_state: AgentState = {
        "plan": pseudo_plan,
        "multi_hop_plan": multi_hop_plan,
        "hop_count": hop_count,
        "workflow_status": workflow_status,
        "step_results": {step["id"]: step},
        "agent_outputs": {ROUTE_MULTI_HOP_AGENT: answer},
        "debug_info": {ROUTE_MULTI_HOP_AGENT: {MULTI_HOP_DEBUG_KEY: debug_payload}},
    }
    if answer:
        next_state["answer"] = answer
    log_node(
        ROUTE_MULTI_HOP_AGENT,
        {**state, **next_state},
        extra={
            "hopCount": hop_count,
            "workflowStatus": workflow_status,
            "degradeReason": debug_payload.get("degrade_reason", ""),
            "baseQuerySource": debug_payload.get("base_query_source", ""),
            "answerPreview": preview(answer, 120),
        },
    )
    return next_state


__all__ = ["multi_hop_node"]

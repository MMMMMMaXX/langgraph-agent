"""Phase 3 PR-3：`multi_hop_node` 单测。

测试策略：
- 不跑真实检索 / LLM：全部通过 monkeypatch 替换 `multi_hop.node` 模块内绑定的
  `decompose_query` / `retrieve_docs_for_rag` / `answer_with_doc_hits` /
  `run_single_hop_retrieval_answer`，由测试决定返回值形状。
- 只断言 PR-3 的核心契约：
  1. State 顶层不泄漏全文，evidence_groups_preview 只在 debug 里且 preview≤120 chars；
  2. `agent_outputs` 只写 `ROUTE_MULTI_HOP_AGENT` 一个 key；
  3. decompose 降级走 `run_single_hop_retrieval_answer` 纯函数，不触 rag_agent_node；
  4. evidence_empty / answer_llm_failed 两条失败路径各自写入 debug.degrade_reason。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.agents.rag.answer_flow import SingleHopAnswerResult
from app.agents.rag.constants import ANSWER_STRATEGY_MULTI_HOP
from app.agents.rag.multi_hop import node as mh_node
from app.agents.rag.multi_hop.types import DecomposeResult, Subquery
from app.constants.multi_hop import (
    DEGRADE_REASON_ANSWER_LLM_FAILED,
    DEGRADE_REASON_BUDGET_EXCEEDED,
    DEGRADE_REASON_EVIDENCE_EMPTY,
    EVIDENCE_PREVIEW_MAX_CHARS,
    MAX_HOPS,
)
from app.constants.routes import ROUTE_MULTI_HOP_AGENT
from app.constants.workflow import (
    STEP_STATUS_FAILED,
    STEP_STATUS_SUCCEEDED,
    TASK_TYPE_MULTI_HOP_RAG,
    WORKFLOW_STATUS_FAILED,
    WORKFLOW_STATUS_PARTIAL,
    WORKFLOW_STATUS_SUCCEEDED,
)

# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------


@dataclass
class _DocRetrievalStub:
    """mimic `DocRetrievalResult`：仅给 multi_hop_node 实际读到的字段。"""

    merged_doc_hits: list[dict]
    retrieval_debug: dict
    errors: list[str]
    timings_ms: dict[str, float]


def _make_doc_hit(
    *,
    doc_id: str,
    chunk_id: str,
    score: float,
    content: str,
) -> dict:
    # `get_chunk_identifier` 读的是 `id` 字段（citations.py:18）；这里统一把
    # 入参的 chunk_id 写到 id 上，避免 dedup 时拿到空串全被丢掉。
    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "score": score,
        "content": content,
    }


def _state(
    *,
    message: str = "跨项目排查登录失败问题",
    rewritten: str | None = None,
) -> dict:
    state: dict[str, Any] = {
        "messages": [{"role": "user", "content": message}],
    }
    if rewritten is not None:
        state["rewritten_query"] = rewritten
    return state


@pytest.fixture
def patch_node(monkeypatch: pytest.MonkeyPatch):
    """统一入口：返回 setter 以便各用例定制各层返回值。"""

    holder: dict[str, Any] = {
        "decompose": None,
        "retrieve": None,
        "answer": None,
        "fallback": None,
    }

    def _decompose(*, rewritten_query: str, role: str = "user") -> DecomposeResult:
        fn = holder["decompose"]
        assert fn is not None, "decompose stub not configured"
        return fn(rewritten_query, role)

    def _retrieve(query: str, **kwargs: Any) -> _DocRetrievalStub:
        fn = holder["retrieve"]
        assert fn is not None, "retrieve stub not configured"
        # 让测试 fn 选择是否消费 kwargs（first-hop 无 refine 参数；refine hop 有）
        return fn(query, **kwargs) if holder.get("retrieve_uses_kwargs") else fn(query)

    def _answer(**kwargs: Any):
        fn = holder["answer"]
        assert fn is not None, "answer stub not configured"
        return fn(**kwargs)

    def _fallback(*, base_query: str, on_delta: Any = None) -> SingleHopAnswerResult:
        fn = holder["fallback"]
        assert fn is not None, "fallback stub not configured"
        return fn(base_query)

    monkeypatch.setattr(mh_node, "decompose_query", _decompose)
    monkeypatch.setattr(mh_node, "retrieve_docs_for_rag", _retrieve)
    monkeypatch.setattr(mh_node, "answer_with_doc_hits", _answer)
    monkeypatch.setattr(mh_node, "run_single_hop_retrieval_answer", _fallback)
    # 流式不影响断言；替换成 no-op 避免依赖 session/token 上下文
    monkeypatch.setattr(
        mh_node,
        "build_answer_streamer",
        lambda state, route: (None, {"used": False}),
    )
    return holder


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_multi_hop_node_success_writes_minimal_state(patch_node) -> None:
    patch_node["decompose"] = lambda q, role: DecomposeResult(
        subqueries=(
            Subquery(id="sq1", intent="entity_lookup", query="A 系统定义"),
            Subquery(id="sq2", intent="procedure", query="B 系统排查步骤"),
        ),
        degraded_to_single_hop=False,
    )

    def _retrieve_stub(query: str) -> _DocRetrievalStub:
        # 不同 sq 返回不同 chunk；每 sq 两条，共 4 条
        return _DocRetrievalStub(
            merged_doc_hits=[
                _make_doc_hit(
                    doc_id=f"doc-{query[:1]}",
                    chunk_id=f"{query[:2]}-c1",
                    score=0.9,
                    content="超长" * 200,  # 确保会被截断
                ),
                _make_doc_hit(
                    doc_id=f"doc-{query[:1]}",
                    chunk_id=f"{query[:2]}-c2",
                    score=0.7,
                    content="备用内容",
                ),
            ],
            retrieval_debug={"refine_docs_dropped": 0},
            errors=[],
            timings_ms={},
        )

    patch_node["retrieve"] = _retrieve_stub
    patch_node["answer"] = lambda **kw: (
        "merged answer",
        [{"ref": "[1]", "doc_id": "doc-A"}],
        {"name": "multi_hop"},
        {},
        [],
        5.0,
    )

    result = mh_node.multi_hop_node(_state(rewritten="跨项目排查登录失败问题"))

    # 硬契约 1：agent_outputs 仅一个 key
    assert list(result["agent_outputs"].keys()) == [ROUTE_MULTI_HOP_AGENT]
    assert result["agent_outputs"][ROUTE_MULTI_HOP_AGENT] == "merged answer"

    # 硬契约 2：step_results 使用 mh1 id
    step = result["step_results"]["mh1"]
    assert step["status"] == STEP_STATUS_SUCCEEDED
    assert step["output"] == "merged answer"
    assert step["meta"]["hop_count"] == 1
    assert "degrade_reason" not in step["meta"]

    # 硬契约 3：multi_hop_plan 只含元数据
    plan = result["multi_hop_plan"]
    assert plan["task_type"] == TASK_TYPE_MULTI_HOP_RAG
    assert plan["degraded_to_single_hop"] is False
    assert [sq["id"] for sq in plan["subqueries"]] == ["sq1", "sq2"]
    for sq in plan["subqueries"]:
        assert set(sq.keys()) == {"id", "intent", "query"}

    # 硬契约 4：workflow_status / hop_count
    assert result["workflow_status"] == WORKFLOW_STATUS_SUCCEEDED
    assert result["hop_count"] == 1

    # 硬契约 5：debug 下 evidence_preview 必须 ≤ 120 chars，且顶层没有全文字段
    debug = result["debug_info"][ROUTE_MULTI_HOP_AGENT]["multi_hop"]
    assert debug["answer_strategy"] == "multi_hop"
    assert debug["base_query_source"] == "rewritten"
    groups = debug["evidence_groups_preview"]
    assert len(groups) == 2
    for group in groups:
        for chunk in group["chunks"]:
            assert len(chunk["preview"]) <= EVIDENCE_PREVIEW_MAX_CHARS
    # State 顶层不得出现 full_chunks / evidence_groups
    for leaked in ("full_chunks", "evidence_groups", "doc_hits"):
        assert leaked not in result


def test_multi_hop_node_tracks_latest_message_when_rewrite_missing(
    patch_node,
) -> None:
    patch_node["decompose"] = lambda q, role: DecomposeResult(
        subqueries=(Subquery(id="sq1", intent="definition", query=q),),
        degraded_to_single_hop=False,
    )
    patch_node["retrieve"] = lambda q: _DocRetrievalStub(
        merged_doc_hits=[
            _make_doc_hit(doc_id="d1", chunk_id="c1", score=1.0, content="内容")
        ],
        retrieval_debug={},
        errors=[],
        timings_ms={},
    )
    patch_node["answer"] = lambda **kw: ("ans", [], {"name": "multi_hop"}, {}, [], 1.0)

    result = mh_node.multi_hop_node(_state(message="hello multi-hop"))
    debug = result["debug_info"][ROUTE_MULTI_HOP_AGENT]["multi_hop"]
    assert debug["base_query_source"] == "latest_user_message"


# ---------------------------------------------------------------------------
# decompose 降级 → 纯函数 fallback
# ---------------------------------------------------------------------------


def test_multi_hop_node_degraded_uses_single_hop_fallback(patch_node) -> None:
    fallback_called: dict[str, Any] = {}

    def _fallback(base_query: str) -> SingleHopAnswerResult:
        fallback_called["base_query"] = base_query
        return SingleHopAnswerResult(
            answer="fallback ans",
            citations=[{"ref": "[1]"}],
            doc_hits=[{"chunk_id": "c1"}],
            merged_doc_hits=[{"chunk_id": "c1"}],
            retrieval_debug={},
            answer_strategy={"name": "doc_default"},
            errors=[],
            timings_ms={"singleHopTotal": 12.0},
        )

    patch_node["decompose"] = lambda q, role: DecomposeResult(
        subqueries=(Subquery(id="sq1", intent="entity_lookup", query=q),),
        degraded_to_single_hop=True,
        error_code="decompose_failed",
    )
    patch_node["fallback"] = _fallback
    # retrieve / answer 不应被调用；给个会抛异常的桩来锁死
    patch_node["retrieve"] = lambda q: (_ for _ in ()).throw(
        AssertionError("retrieve must not be called in fallback branch")
    )
    patch_node["answer"] = lambda **kw: (_ for _ in ()).throw(
        AssertionError("answer must not be called in fallback branch")
    )

    result = mh_node.multi_hop_node(_state(rewritten="跨项目排查登录失败问题"))

    assert fallback_called["base_query"] == "跨项目排查登录失败问题"
    assert result["agent_outputs"] == {ROUTE_MULTI_HOP_AGENT: "fallback ans"}

    step = result["step_results"]["mh1"]
    assert step["status"] == STEP_STATUS_SUCCEEDED
    assert step["meta"]["single_hop_fallback"] is True
    assert step["meta"]["degrade_reason"] == "decompose_failed"

    debug = result["debug_info"][ROUTE_MULTI_HOP_AGENT]["multi_hop"]
    assert debug["single_hop_fallback"] is True
    assert debug["degrade_reason"] == "decompose_failed"


# ---------------------------------------------------------------------------
# 失败路径
# ---------------------------------------------------------------------------


def test_multi_hop_node_empty_evidence_marks_failed(patch_node) -> None:
    patch_node["decompose"] = lambda q, role: DecomposeResult(
        subqueries=(Subquery(id="sq1", intent="entity_lookup", query=q),),
        degraded_to_single_hop=False,
    )
    patch_node["retrieve"] = lambda q: _DocRetrievalStub(
        merged_doc_hits=[],
        retrieval_debug={},
        errors=[],
        timings_ms={},
    )
    # answer 不应被调；用异常桩
    patch_node["answer"] = lambda **kw: (_ for _ in ()).throw(
        AssertionError("answer must not be called on empty evidence")
    )

    result = mh_node.multi_hop_node(_state(rewritten="跨项目登录失败"))

    step = result["step_results"]["mh1"]
    assert step["status"] == STEP_STATUS_FAILED
    assert step["meta"]["degrade_reason"] == DEGRADE_REASON_EVIDENCE_EMPTY
    assert result["agent_outputs"][ROUTE_MULTI_HOP_AGENT] == ""
    assert result["workflow_status"] == WORKFLOW_STATUS_FAILED
    assert "answer" not in result


def test_multi_hop_node_answer_llm_exception_marks_failed(patch_node) -> None:
    patch_node["decompose"] = lambda q, role: DecomposeResult(
        subqueries=(Subquery(id="sq1", intent="entity_lookup", query=q),),
        degraded_to_single_hop=False,
    )
    patch_node["retrieve"] = lambda q: _DocRetrievalStub(
        merged_doc_hits=[
            _make_doc_hit(doc_id="d1", chunk_id="c1", score=0.9, content="evidence")
        ],
        retrieval_debug={},
        errors=[],
        timings_ms={},
    )

    def _boom(**kwargs: Any):
        raise RuntimeError("LLM down")

    patch_node["answer"] = _boom

    result = mh_node.multi_hop_node(_state(rewritten="跨项目登录失败"))

    step = result["step_results"]["mh1"]
    assert step["status"] == STEP_STATUS_FAILED
    assert step["meta"]["degrade_reason"] == DEGRADE_REASON_ANSWER_LLM_FAILED
    debug = result["debug_info"][ROUTE_MULTI_HOP_AGENT]["multi_hop"]
    assert debug["degrade_reason"] == DEGRADE_REASON_ANSWER_LLM_FAILED
    assert debug.get("errors"), "LLM 失败必须写到 debug.errors"
    assert result["workflow_status"] == WORKFLOW_STATUS_FAILED


def test_multi_hop_node_empty_base_query_fails_fast(patch_node) -> None:
    # 不给任何消息：base_query 空 → 直接失败，decompose/retrieve/answer 全都不应被调
    patch_node["decompose"] = lambda q, role: (_ for _ in ()).throw(
        AssertionError("decompose must not be called on empty query")
    )
    patch_node["retrieve"] = lambda q: (_ for _ in ()).throw(AssertionError("x"))
    patch_node["answer"] = lambda **kw: (_ for _ in ()).throw(AssertionError("x"))

    result = mh_node.multi_hop_node({"messages": []})

    step = result["step_results"]["mh1"]
    assert step["status"] == STEP_STATUS_FAILED
    assert step["meta"]["degrade_reason"] == DEGRADE_REASON_EVIDENCE_EMPTY
    assert result["hop_count"] == 0


# ---------------------------------------------------------------------------
# Refine loop / budget / strategy 契约
# ---------------------------------------------------------------------------


def test_multi_hop_node_triggers_refine_second_hop_on_insufficient_chunks(
    patch_node,
) -> None:
    """首跳单 chunk (< MIN_CHUNKS_PER_SUBQUERY=2) → gap detector 要求 refine。

    断言：
    - 同一个 sq 被 retrieve 了两次（一次首跳，一次 refine）
    - 第二次调用带上了 top_k_multiplier > 1（insufficient_chunks 映射）
    - hop_count = 2
    """

    retrieve_calls: list[dict] = []

    def _retrieve_stub(query: str, **kwargs: Any) -> _DocRetrievalStub:
        retrieve_calls.append({"query": query, "kwargs": kwargs})
        if len(retrieve_calls) == 1:
            # 首跳：只给 1 个 chunk（MIN_CHUNKS_PER_SUBQUERY=2 要求 ≥ 2）
            return _DocRetrievalStub(
                merged_doc_hits=[
                    _make_doc_hit(doc_id="d1", chunk_id="c1", score=0.9, content="部分")
                ],
                retrieval_debug={},
                errors=[],
                timings_ms={},
            )
        # Refine hop：补 1 个新 chunk → 累计达标
        return _DocRetrievalStub(
            merged_doc_hits=[
                _make_doc_hit(doc_id="d2", chunk_id="c2", score=0.8, content="补充")
            ],
            retrieval_debug={},
            errors=[],
            timings_ms={},
        )

    patch_node["retrieve_uses_kwargs"] = True
    patch_node["decompose"] = lambda q, role: DecomposeResult(
        subqueries=(Subquery(id="sq1", intent="entity_lookup", query="A 系统"),),
        degraded_to_single_hop=False,
    )
    patch_node["retrieve"] = _retrieve_stub
    patch_node["answer"] = lambda **kw: (
        "ans",
        [{"ref": "[1]"}],
        kw["strategy"],
        {},
        [],
        1.0,
    )

    result = mh_node.multi_hop_node(_state(rewritten="跨项目排查 A"))

    assert (
        len(retrieve_calls) == 2
    ), f"expected 2 retrievals (first-hop + refine), got {len(retrieve_calls)}"
    # 首跳不带 refine 参数
    assert retrieve_calls[0]["kwargs"] == {}
    # 第二跳应携带 insufficient_chunks 翻译出的 top_k_multiplier > 1
    refine_kwargs = retrieve_calls[1]["kwargs"]
    assert refine_kwargs.get("top_k_multiplier", 1.0) > 1.0
    # refine 触达 retrieve_docs_for_rag 的完整结构化参数（证明走 PR-2 接口）
    for key in (
        "top_k_multiplier",
        "rerank_top_k_multiplier",
        "entity_hints",
        "per_doc_limit",
        "force_source_diversity",
    ):
        assert key in refine_kwargs

    assert result["hop_count"] == 2
    assert result["workflow_status"] == WORKFLOW_STATUS_SUCCEEDED


def test_multi_hop_node_budget_exceeded_marks_partial(patch_node) -> None:
    """每跳只回一个 chunk 且 id 相同 → gap 永远闭不上，hop 耗尽。

    断言：
    - hop_count == MAX_HOPS
    - step.status=SUCCEEDED（有答案）但 degrade_reason=budget_exceeded
    - workflow_status = PARTIAL（有证据，不算 FAIL）
    """

    def _retrieve_stub(query: str, **kwargs: Any) -> _DocRetrievalStub:
        # 永远只给这 1 个 chunk；insufficient_chunks 每跳都触发
        return _DocRetrievalStub(
            merged_doc_hits=[
                _make_doc_hit(doc_id="d1", chunk_id="c1", score=0.9, content="仅此")
            ],
            retrieval_debug={},
            errors=[],
            timings_ms={},
        )

    patch_node["retrieve_uses_kwargs"] = True
    patch_node["decompose"] = lambda q, role: DecomposeResult(
        subqueries=(Subquery(id="sq1", intent="entity_lookup", query=q),),
        degraded_to_single_hop=False,
    )
    patch_node["retrieve"] = _retrieve_stub
    patch_node["answer"] = lambda **kw: (
        "partial ans",
        [{"ref": "[1]"}],
        kw["strategy"],
        {},
        [],
        1.0,
    )

    result = mh_node.multi_hop_node(_state(rewritten="跨项目查询"))

    assert result["hop_count"] == MAX_HOPS
    assert result["workflow_status"] == WORKFLOW_STATUS_PARTIAL

    step = result["step_results"]["mh1"]
    assert step["status"] == STEP_STATUS_SUCCEEDED
    assert step["output"] == "partial ans"
    assert step["meta"]["degrade_reason"] == DEGRADE_REASON_BUDGET_EXCEEDED

    debug = result["debug_info"][ROUTE_MULTI_HOP_AGENT]["multi_hop"]
    assert debug["degrade_reason"] == DEGRADE_REASON_BUDGET_EXCEEDED
    assert debug["refine_loop"]["final_plan_ok"] is False


def test_multi_hop_node_answer_uses_multi_hop_strategy(patch_node) -> None:
    """正常多跳回答必须走 ANSWER_STRATEGY_MULTI_HOP，而不是被 query_type 覆盖。"""

    captured: dict[str, Any] = {}

    def _answer_stub(**kwargs: Any):
        captured.update(kwargs)
        return (
            "ans",
            [],
            kwargs["strategy"],
            {},
            [],
            1.0,
        )

    patch_node["decompose"] = lambda q, role: DecomposeResult(
        subqueries=(
            Subquery(id="sq1", intent="entity_lookup", query="A"),
            Subquery(id="sq2", intent="procedure", query="B"),
        ),
        degraded_to_single_hop=False,
    )
    patch_node["retrieve"] = lambda q: _DocRetrievalStub(
        merged_doc_hits=[
            _make_doc_hit(doc_id=q, chunk_id=f"{q}-1", score=0.9, content="x"),
            _make_doc_hit(doc_id=q, chunk_id=f"{q}-2", score=0.8, content="y"),
        ],
        retrieval_debug={},
        errors=[],
        timings_ms={},
    )
    patch_node["answer"] = _answer_stub

    result = mh_node.multi_hop_node(_state(rewritten="跨项目汇总"))

    # answer_with_doc_hits 必须用 multi_hop 策略；不走 query_type 驱动的单跳策略
    assert captured["query_type"] == ""
    assert captured["strategy"]["name"] == ANSWER_STRATEGY_MULTI_HOP

    # 同时 debug 层面确认 strategy name 被透出来，方便 eval 统计
    debug = result["debug_info"][ROUTE_MULTI_HOP_AGENT]["multi_hop"]
    assert debug["answer_strategy"] == ANSWER_STRATEGY_MULTI_HOP


def test_multi_hop_node_global_source_diversity_refine(patch_node) -> None:
    """两条 comparison_arm 首跳都命中同一个 d1 → global_no_source_diversity。

    gap 检测器此时只往 per_sq 上写结构化参数（exclude_doc_ids / per_doc_limit /
    diversity_by_doc），不追加 missing_aspects。如果 refine loop 只看
    missing_aspects 是否非空，就会跳过这轮 refine，空转到 MAX_HOPS。

    断言：
    - 两个 sq 各自被 retrieve 两次（首跳 + diversity refine）
    - 二跳每条 refine 调用都带上 exclude_doc_ids={"d1"}, per_doc_limit=1,
      force_source_diversity=True
    - 二跳真的引入了新 doc（d2/d3）→ global 达标 → workflow SUCCEEDED
    """

    retrieve_calls: list[dict] = []

    def _retrieve_stub(query: str, **kwargs: Any) -> _DocRetrievalStub:
        retrieve_calls.append({"query": query, "kwargs": kwargs})
        # 首跳：两条 sq 都从 d1 取 2 个 chunk（per_sq 达标，但 doc 只有一个）
        if not kwargs:
            suffix = "A" if "A" in query else "B"
            return _DocRetrievalStub(
                merged_doc_hits=[
                    _make_doc_hit(
                        doc_id="d1",
                        chunk_id=f"d1-{suffix}-1",
                        score=0.9,
                        content="首跳同源证据一",
                    ),
                    _make_doc_hit(
                        doc_id="d1",
                        chunk_id=f"d1-{suffix}-2",
                        score=0.85,
                        content="首跳同源证据二",
                    ),
                ],
                retrieval_debug={},
                errors=[],
                timings_ms={},
            )
        # Refine hop：携带 diversity 参数 → 从新 doc 补位
        suffix = "A" if "A" in query else "B"
        new_doc = "d2" if suffix == "A" else "d3"
        return _DocRetrievalStub(
            merged_doc_hits=[
                _make_doc_hit(
                    doc_id=new_doc,
                    chunk_id=f"{new_doc}-{suffix}-1",
                    score=0.8,
                    content="多样性补位",
                ),
            ],
            retrieval_debug={},
            errors=[],
            timings_ms={},
        )

    patch_node["retrieve_uses_kwargs"] = True
    patch_node["decompose"] = lambda q, role: DecomposeResult(
        subqueries=(
            Subquery(id="sq1", intent="comparison_arm", query="A 方案"),
            Subquery(id="sq2", intent="comparison_arm", query="B 方案"),
        ),
        degraded_to_single_hop=False,
    )
    patch_node["retrieve"] = _retrieve_stub
    patch_node["answer"] = lambda **kw: (
        "ans",
        [{"ref": "[1]"}],
        kw["strategy"],
        {},
        [],
        1.0,
    )

    result = mh_node.multi_hop_node(_state(rewritten="对比 A 方案和 B 方案"))

    # 首跳两次 + refine 两次 = 4 次 retrieve
    assert (
        len(retrieve_calls) == 4
    ), f"expected 4 retrievals (2 sq × 2 hops), got {len(retrieve_calls)}"
    # 前两次是首跳，无 kwargs
    assert retrieve_calls[0]["kwargs"] == {}
    assert retrieve_calls[1]["kwargs"] == {}

    # 后两次是 diversity refine：exclude_doc_ids 包含 d1，强制 per_doc_limit=1 + 多样性
    for refine_call in retrieve_calls[2:]:
        kwargs = refine_call["kwargs"]
        assert (
            "d1" in kwargs["exclude_doc_ids"]
        ), f"diversity refine 必须排除已覆盖的 d1，got {kwargs.get('exclude_doc_ids')}"
        assert kwargs["per_doc_limit"] == 1
        assert kwargs["force_source_diversity"] is True

    # hop 在第二跳关闭（不应空转到 MAX_HOPS）
    assert result["hop_count"] == 2
    assert result["workflow_status"] == WORKFLOW_STATUS_SUCCEEDED


# ---------------------------------------------------------------------------
# contract: multi_hop_node → Composer / Verifier
# ---------------------------------------------------------------------------


def test_multi_hop_node_output_flows_through_composer_and_verifier(
    patch_node,
) -> None:
    """契约测试：multi_hop_node 的 State 能被 Composer / Verifier 直接消费。

    multi_hop_node 只写 `plan`/`multi_hop_plan`/`step_results["mh1"]`；Composer
    必须命中 multi-hop 直通分支（不回落 plan_failed），Verifier 必须命中
    coverage 检查（不被 empty-plan early-return 截断）。
    """

    from app.agents.composer_agent import composer_node
    from app.agents.verifier_agent import verifier_node
    from app.auth.context import AuthContext
    from app.constants.multi_hop import (
        MULTI_HOP_STEP_ID,
        RISK_WARN_MULTI_HOP_COVERAGE,
    )
    from app.constants.workflow import (
        COMPOSER_FALLBACK_PLAN_FAILED,
        COMPOSER_OUTPUT_KEY,
        TASK_TYPE_MULTI_HOP_RAG,
    )

    patch_node["decompose"] = lambda q, role: DecomposeResult(
        subqueries=(
            Subquery(id="sq1", intent="entity_lookup", query="A 定义"),
            Subquery(id="sq2", intent="procedure", query="B 排查"),
        ),
        degraded_to_single_hop=False,
    )

    # sq2 召回为空 → missing_coverage_sq_ids 会登记 sq2
    def _retrieve_stub(query: str) -> _DocRetrievalStub:
        if query.startswith("A"):
            return _DocRetrievalStub(
                merged_doc_hits=[
                    _make_doc_hit(
                        doc_id="doc-A", chunk_id="A-c1", score=0.9, content="内容"
                    ),
                    _make_doc_hit(
                        doc_id="doc-A", chunk_id="A-c2", score=0.7, content="内容"
                    ),
                ],
                retrieval_debug={},
                errors=[],
                timings_ms={},
            )
        return _DocRetrievalStub(
            merged_doc_hits=[], retrieval_debug={}, errors=[], timings_ms={}
        )

    patch_node["retrieve"] = _retrieve_stub
    patch_node["answer"] = lambda **kw: (
        "multi-hop 综合答案",
        [{"ref": "[1]", "doc_id": "doc-A", "chunk_id": "A-c1"}],
        {"name": "multi_hop"},
        {},
        [],
        5.0,
    )

    mh_state = mh_node.multi_hop_node(_state(rewritten="跨项目排查"))

    # multi_hop_node 契约：写 plan + multi_hop_plan，pseudo plan 有 mh1 一条 step
    assert mh_state["plan"]["task_type"] == TASK_TYPE_MULTI_HOP_RAG
    assert [s["id"] for s in mh_state["plan"]["steps"]] == [MULTI_HOP_STEP_ID]

    # Composer：命中直通分支，answer 原样透传
    auth = AuthContext(tenant_id="t1", user_id="u1", role="user", anonymous=False)
    composer_in = {
        **mh_state,
        "messages": [{"role": "user", "content": "跨项目排查"}],
        "auth": auth,
        "request_id": "req-x",
        "session_id": "sess-x",
        "plan_id": "plan-x",
        "verification": {
            "status": "pass",
            "missing_fields": [],
            "unsupported_claims": [],
            "risk_warnings": [],
        },
        "pending_confirmation": {},
    }
    composer_out = composer_node(composer_in)
    assert composer_out["answer"].startswith("multi-hop 综合答案")
    assert COMPOSER_FALLBACK_PLAN_FAILED not in composer_out["answer"]
    debug = composer_out["debug_info"][list(composer_out["debug_info"].keys())[0]]
    assert debug.get("multi_hop_passthrough") is True
    composer_output = composer_out["agent_outputs"][COMPOSER_OUTPUT_KEY]
    assert [a["step"] for a in composer_output["completed_actions"]] == [
        MULTI_HOP_STEP_ID
    ]

    # Verifier：empty-plan early-return 已绕过，coverage 风险码被加上
    verifier_in = {**composer_in, "verification": {}}
    verifier_out = verifier_node(verifier_in)
    assert RISK_WARN_MULTI_HOP_COVERAGE in verifier_out["verification"]["risk_warnings"]

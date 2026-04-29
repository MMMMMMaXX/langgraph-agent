"""RAG pipeline 单测：doc_pipeline + memory_pipeline + rewrite。

E2E 测试把这三个 pipeline 整个 stub 掉了，内部分支全没覆盖。这里把 I/O
（Chroma 查询、rerank LLM、rewrite LLM）打到 pipeline 模块命名空间上，专门
驱动内部分支。
"""

from __future__ import annotations

import pytest

from app.agents.rag.constants import (
    ANSWER_TOKENS_MIN,
    QUERY_TYPE_COMPARISON,
    QUERY_TYPE_DEFINITION,
    QUERY_TYPE_FALLBACK,
    QUERY_TYPE_FOLLOWUP,
)
from app.agents.rag.doc_pipeline import (
    build_query_type_hybrid_weights,
    retrieve_docs_for_rag,
    select_source_diverse_hits,
)
from app.agents.rag.memory_pipeline import filter_memory_hits, retrieve_memory_for_rag
from app.agents.rag.rewrite import (
    get_user_messages,
    rewrite_rag_query,
)
from app.agents.rag.strategy import adapt_strategy_max_tokens

# ---------------------------------------------------------------------------
# rewrite.get_user_messages （纯函数）
# ---------------------------------------------------------------------------


def test_get_user_messages_filters_and_limits() -> None:
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ]
    # 默认 limit=2，取最后两条 user
    result = get_user_messages(messages)
    assert [m["content"] for m in result] == ["u2", "u3"]

    # limit=1
    assert [m["content"] for m in get_user_messages(messages, limit=1)] == ["u3"]


def test_get_user_messages_empty_returns_empty() -> None:
    assert get_user_messages([]) == []


def test_get_user_messages_no_user_role_returns_empty() -> None:
    messages = [{"role": "assistant", "content": "a"}]
    assert get_user_messages(messages) == []


# ---------------------------------------------------------------------------
# rewrite.rewrite_rag_query （走 LLM：llm_stub 拦截 _create_chat_completion）
# ---------------------------------------------------------------------------


def test_rewrite_rag_query_short_followup_without_context_skips_llm(llm_stub) -> None:
    """没有上下文时，短追问不调 LLM，只做问号归一化。"""
    result = rewrite_rag_query("那北京呢", messages=[], summary="")
    assert result.query == "那北京呢？"
    assert result.errors == []
    assert result.timing_ms >= 0
    assert result.mode == "skip"
    assert result.skipped_reason == "no_context"
    assert llm_stub.calls == []


def test_rewrite_rag_query_na_prefix_triggers_llm(llm_stub) -> None:
    """有上下文的追问走 LLM 改写路径。"""
    llm_stub.set_response("北京今天气温多少度")

    result = rewrite_rag_query(
        "那首都现在情况如何",
        messages=[{"role": "user", "content": "上海气温"}],
        summary="用户关注各城市气温",
    )
    assert result.query == "北京今天气温多少度？"
    assert result.errors == []
    assert result.mode == "llm"
    assert result.trigger == "followup_query"
    # 确实调用了 LLM
    assert len(llm_stub.calls) == 1
    call = llm_stub.calls[0]
    # summary 应作为 system 消息塞进 prompt
    sent_messages = call["messages"]
    assert any(
        m["role"] == "system" and "用户关注各城市气温" in m["content"]
        for m in sent_messages
    )


def test_rewrite_rag_query_llm_failure_falls_back_to_append_mark(llm_stub) -> None:
    """LLM 改写失败时降级为原句加"？"，错误记录到 errors。"""

    def raise_error(**_):
        raise RuntimeError("provider down")

    llm_stub.set_response_fn(raise_error)

    result = rewrite_rag_query(
        "那上海怎么样",
        messages=[
            {"role": "user", "content": "北京气候怎么样？"},
            {"role": "assistant", "content": "北京气候四季分明。"},
            {"role": "user", "content": "那上海怎么样"},
        ],
        summary="用户在比较城市气候",
    )
    # 原 message 不以问号结尾 → 补上
    assert result.query.endswith("？")
    # 错误信息被记录
    assert len(result.errors) == 1


def test_rewrite_rag_query_default_branch_appends_question_mark(llm_stub) -> None:
    """非追问默认直接补问号，不调用 LLM。"""
    result = rewrite_rag_query("深圳下周空气质量", messages=[], summary="")
    assert result.query == "深圳下周空气质量？"
    assert result.errors == []
    # 这条路径不调 LLM
    assert llm_stub.calls == []


def test_rewrite_rag_query_preserves_existing_question_mark(llm_stub) -> None:
    """已带问号的原句不会被二次添加。"""
    result = rewrite_rag_query("深圳下周空气质量？", messages=[], summary="")
    assert result.query == "深圳下周空气质量？"


# ---------------------------------------------------------------------------
# doc_pipeline.retrieve_docs_for_rag
# ---------------------------------------------------------------------------


def _doc(
    score: float, doc_id: str = "d1", chunk_index: int = 0, **extra: object
) -> dict:
    base = {
        "id": f"{doc_id}-{chunk_index}",
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "start_char": 0,
        "end_char": 10,
        "content": "text",
        "score": score,
        "semantic_score": score,
        "keyword_score_norm": score,
        "retrieval_sources": ["dense", "keyword"],
    }
    base.update(extra)
    return base


@pytest.fixture
def doc_pipeline_io(monkeypatch: pytest.MonkeyPatch) -> dict:
    """打桩 doc_pipeline 依赖的 dense / lexical / rerank。

    返回 harness，测试里改 harness["docs"] / harness["lexical_docs"] 即可切换场景。
    `docs` 保留为 dense 默认输入，兼容早期测试命名。
    """
    import app.agents.rag.doc_pipeline as dp_mod

    harness: dict = {
        "docs": [],
        "lexical_docs": [],
        "dense_error": None,
        "lexical_error": None,
        "rerank_fn": lambda query, hits, top_k: hits[:top_k],
        "dense_calls": 0,
        "lexical_calls": 0,
        "rerank_calls": 0,
    }

    def fake_dense_retrieve_docs(query: str, top_k: int) -> list[dict]:
        harness["dense_calls"] += 1
        if harness["dense_error"] is not None:
            raise harness["dense_error"]
        hits = [dict(hit) for hit in harness["docs"]]
        for hit in hits:
            hit.setdefault("retrieval_source", "dense")
        return hits

    def fake_keyword_retrieve_docs(query: str, top_k: int) -> list[dict]:
        harness["lexical_calls"] += 1
        if harness["lexical_error"] is not None:
            raise harness["lexical_error"]
        hits = [dict(hit) for hit in harness["lexical_docs"]]
        for hit in hits:
            hit.setdefault("retrieval_source", "keyword")
        return hits

    def fake_rerank(query: str, hits: list, top_k: int = 2) -> list:
        harness["rerank_calls"] += 1
        return harness["rerank_fn"](query, hits, top_k)

    monkeypatch.setattr(dp_mod, "dense_retrieve_docs", fake_dense_retrieve_docs)
    monkeypatch.setattr(dp_mod, "keyword_retrieve_docs", fake_keyword_retrieve_docs)
    monkeypatch.setattr(dp_mod, "rerank", fake_rerank)
    return harness


def test_build_query_type_hybrid_weights_are_centralized() -> None:
    # confidence=1.0（默认）时，与原始 type-specific 权重完全一致
    assert build_query_type_hybrid_weights(QUERY_TYPE_DEFINITION) == (0.55, 0.45)
    assert build_query_type_hybrid_weights(QUERY_TYPE_COMPARISON) == (0.6, 0.4)
    assert build_query_type_hybrid_weights(QUERY_TYPE_FOLLOWUP) == (0.7, 0.3)
    assert build_query_type_hybrid_weights(QUERY_TYPE_FALLBACK) == (0.5, 0.5)
    assert build_query_type_hybrid_weights("unknown") == (0.65, 0.35)


def test_build_query_type_hybrid_weights_confidence_decay() -> None:
    # FACTUAL（base 0.65/0.35）在 confidence=0.6 时应向 balanced 收敛
    # 公式：alpha = 0.65 * 0.6 + 0.5 * 0.4 = 0.59
    alpha, beta = build_query_type_hybrid_weights("unknown", confidence=0.6)
    assert alpha == 0.59
    assert beta == 0.41

    # 高置信度（0.9）对高 alpha 类型影响很小
    # alpha = 0.7 * 0.9 + 0.5 * 0.1 = 0.68
    alpha2, beta2 = build_query_type_hybrid_weights(QUERY_TYPE_FOLLOWUP, confidence=0.9)
    assert alpha2 == 0.68
    assert beta2 == 0.32

    # confidence=0.5 → 完全均衡
    alpha3, beta3 = build_query_type_hybrid_weights(
        QUERY_TYPE_DEFINITION, confidence=0.5
    )
    assert alpha3 == 0.525
    assert beta3 == 0.475


def test_doc_pipeline_search_error_recorded(doc_pipeline_io: dict) -> None:
    doc_pipeline_io["dense_error"] = RuntimeError("chroma down")
    result = retrieve_docs_for_rag("天气")

    assert result.docs == []
    assert result.filtered_docs == []
    assert result.doc_hits == []
    assert len(result.errors) == 1
    assert "docSearch" in result.timings_ms
    assert "docDenseSearch" in result.timings_ms
    assert "docLexicalSearch" in result.timings_ms
    assert "docHybridMerge" in result.timings_ms
    # 搜都搜不到，rerank 不该被调
    assert doc_pipeline_io["rerank_calls"] == 0


def test_doc_pipeline_empty_docs_returns_empty_result(doc_pipeline_io: dict) -> None:
    doc_pipeline_io["docs"] = []
    result = retrieve_docs_for_rag("天气")

    assert result.docs == []
    assert result.filtered_docs == []
    assert result.doc_hits == []
    assert result.errors == []
    assert doc_pipeline_io["rerank_calls"] == 0


def test_doc_pipeline_filters_below_threshold(doc_pipeline_io: dict) -> None:
    """score < 0.5 的候选被丢，走不到 rerank。"""
    doc_pipeline_io["docs"] = [
        _doc(score=0.2, doc_id="d1"),
        _doc(score=0.1, doc_id="d2"),
    ]
    result = retrieve_docs_for_rag("q")
    # 都低于 0.5 且也低于 soft_match 0.35 → 全过滤
    assert result.filtered_docs == []


def test_doc_pipeline_soft_match_keeps_top_one(doc_pipeline_io: dict) -> None:
    """都没达到 0.5 硬阈值，但 top1 ≥ 0.35 → 保留 top1。"""
    doc_pipeline_io["docs"] = [
        # dense-only 候选会经过 hybrid_alpha 融合，0.62 * 0.65 ≈ 0.40。
        _doc(score=0.62, doc_id="d1"),  # 最终分介于 0.35 ~ 0.5
        _doc(score=0.1, doc_id="d2"),
    ]
    result = retrieve_docs_for_rag("q")
    # 只保留 top1
    assert len(result.filtered_docs) == 1
    assert result.filtered_docs[0]["doc_id"] == "d1"


def test_doc_pipeline_fallback_query_disables_soft_match(
    doc_pipeline_io: dict,
) -> None:
    """fallback 类问题更保守，低于硬阈值时不靠 soft-match 硬答。"""
    doc_pipeline_io["docs"] = [
        # 非 fallback 时最终 hybrid 分约 0.40，会命中 soft-match。
        _doc(score=0.62, doc_id="d1"),
        _doc(score=0.1, doc_id="d2"),
    ]

    result = retrieve_docs_for_rag("q", query_type=QUERY_TYPE_FALLBACK)

    assert result.filtered_docs == []
    assert result.retrieval_debug["query_type"] == QUERY_TYPE_FALLBACK


def test_doc_pipeline_comparison_soft_match_keeps_multiple_docs(
    doc_pipeline_io: dict,
) -> None:
    """对比类问题在 lexical-only 低分场景下，应保留多个 soft-match 候选。"""
    doc_pipeline_io["docs"] = [
        _doc(score=0.62, doc_id="a", chunk_index=0),
        _doc(score=0.62, doc_id="b", chunk_index=0),
    ]

    result = retrieve_docs_for_rag(
        "A 和 B 有什么区别", query_type=QUERY_TYPE_COMPARISON
    )

    assert [doc["doc_id"] for doc in result.filtered_docs] == ["a", "b"]


def test_doc_pipeline_skips_rerank_when_policy_allows(doc_pipeline_io: dict) -> None:
    """单候选 → should_skip_doc_rerank 返回 True → 不调 rerank。"""
    doc_pipeline_io["docs"] = [_doc(score=0.9, doc_id="d1", chunk_index=0)]
    result = retrieve_docs_for_rag("q")

    assert len(result.doc_hits) == 1
    assert doc_pipeline_io["rerank_calls"] == 0
    assert result.retrieval_debug["rerank_skipped"] is True


def test_doc_pipeline_comparison_query_uses_larger_top_k(
    doc_pipeline_io: dict,
) -> None:
    result = retrieve_docs_for_rag(
        "A 和 B 有什么区别", query_type=QUERY_TYPE_COMPARISON
    )

    assert result.retrieval_debug["query_type"] == QUERY_TYPE_COMPARISON
    assert result.retrieval_debug["requested_top_k"] >= 8
    assert result.retrieval_debug["candidate_top_k"] >= 40
    assert result.retrieval_debug["hybrid_alpha"] == 0.6
    assert result.retrieval_debug["hybrid_beta"] == 0.4
    assert "dense_search" in result.retrieval_debug["pipeline_steps"]
    assert "lexical_search" in result.retrieval_debug["pipeline_steps"]
    assert "hybrid_merge" in result.retrieval_debug["pipeline_steps"]
    assert "threshold" in result.retrieval_debug["pipeline_steps"]
    assert "source_diversity" in result.retrieval_debug["pipeline_steps"]
    assert result.retrieval_debug["source_diversity_enabled"] is True


def test_doc_pipeline_definition_query_boosts_lexical_weight(
    doc_pipeline_io: dict,
) -> None:
    doc_pipeline_io["docs"] = [
        _doc(score=0.8, doc_id="dense", content="semantic only", chunk_index=0),
    ]
    doc_pipeline_io["lexical_docs"] = [
        _doc(
            score=0.0,
            doc_id="lexical",
            content="WAI-ARIA exact match",
            semantic_score=0.0,
            keyword_score=10.0,
            keyword_score_norm=1.0,
            chunk_index=0,
        ),
    ]

    result = retrieve_docs_for_rag("WAI-ARIA 是什么", query_type=QUERY_TYPE_DEFINITION)

    assert result.docs[0]["doc_id"] == "lexical"
    assert result.retrieval_debug["requested_top_k"] >= 6
    assert result.retrieval_debug["hybrid_alpha"] == 0.55
    assert result.retrieval_debug["hybrid_beta"] == 0.45
    assert result.retrieval_debug["hybrid_weight_strategy"] == "query_type_dynamic"


def test_doc_pipeline_followup_query_expands_rerank_top_k(
    doc_pipeline_io: dict,
) -> None:
    result = retrieve_docs_for_rag(
        "Comate Skill 怎么调试？", query_type=QUERY_TYPE_FOLLOWUP
    )

    assert result.retrieval_debug["requested_top_k"] >= 6
    assert result.retrieval_debug["candidate_top_k"] >= 24
    assert result.retrieval_debug["hybrid_alpha"] == 0.7
    assert result.retrieval_debug["hybrid_beta"] == 0.3


def test_doc_pipeline_short_query_expands_candidate_pool(
    doc_pipeline_io: dict,
) -> None:
    result = retrieve_docs_for_rag("WAI-ARIA")

    assert result.retrieval_debug["requested_top_k"] >= 6
    assert result.retrieval_debug["candidate_top_k"] >= 30


def test_doc_pipeline_rerank_called_when_policy_forbids_skip(
    doc_pipeline_io: dict,
) -> None:
    """3 个候选超过跳过阈值，必须走 rerank。"""
    doc_pipeline_io["docs"] = [
        _doc(score=0.9, doc_id="d1", chunk_index=0),
        _doc(score=0.85, doc_id="d2", chunk_index=0),
        _doc(score=0.8, doc_id="d3", chunk_index=0),
    ]
    result = retrieve_docs_for_rag("q")
    assert doc_pipeline_io["rerank_calls"] == 1
    assert result.retrieval_debug["rerank_skipped"] is False


def test_doc_pipeline_rerank_error_falls_back_to_filtered(
    doc_pipeline_io: dict,
) -> None:
    """rerank 抛错时，doc_hits 回退为 filtered_docs 的前 top_k，错误记录下来。"""
    doc_pipeline_io["docs"] = [
        _doc(score=0.9, doc_id="d1", chunk_index=0),
        _doc(score=0.85, doc_id="d2", chunk_index=0),
        _doc(score=0.8, doc_id="d3", chunk_index=0),
    ]

    def raise_error(*_args, **_kwargs):
        raise RuntimeError("rerank crashed")

    doc_pipeline_io["rerank_fn"] = raise_error

    result = retrieve_docs_for_rag("q")
    # 降级为 filtered[:2]（doc_rerank_top_k=2）
    assert len(result.doc_hits) == 2
    assert len(result.errors) == 1


def test_doc_pipeline_lexical_only_hit_can_flow_through(
    doc_pipeline_io: dict,
) -> None:
    """dense 召回为空时，lexical 结果仍能进入 hybrid/threshold/rerank。"""
    doc_pipeline_io["docs"] = []
    doc_pipeline_io["lexical_docs"] = [
        _doc(
            score=0.0,
            doc_id="lex",
            content="WAI-ARIA text",
            keyword_score=3.0,
            keyword_score_norm=1.0,
            semantic_score=0.0,
        )
    ]

    result = retrieve_docs_for_rag("WAI-ARIA")

    assert result.docs[0]["doc_id"] == "lex"
    assert result.retrieval_debug["dense_count"] == 0
    assert result.retrieval_debug["lexical_count"] == 1
    assert result.retrieval_debug["hybrid_count"] == 1


def test_doc_pipeline_hybrid_merge_dedupes_dense_and_lexical(
    doc_pipeline_io: dict,
) -> None:
    """同一个 chunk 被 dense 和 lexical 同时召回时，只保留一条并记录多来源。"""
    dense_doc = _doc(score=0.7, doc_id="same", chunk_index=1)
    lexical_doc = _doc(
        score=0.0,
        doc_id="same",
        chunk_index=1,
        semantic_score=0.0,
        keyword_score=3.0,
        keyword_score_norm=1.0,
        retrieval_sources=["keyword"],
    )
    doc_pipeline_io["docs"] = [dense_doc]
    doc_pipeline_io["lexical_docs"] = [lexical_doc]

    result = retrieve_docs_for_rag("WAI-ARIA")

    assert len(result.docs) == 1
    assert result.docs[0]["doc_id"] == "same"
    assert set(result.docs[0]["retrieval_sources"]) == {"dense", "keyword"}
    assert result.retrieval_debug["dense_count"] == 1
    assert result.retrieval_debug["lexical_count"] == 1
    assert result.retrieval_debug["hybrid_count"] == 1


def test_select_source_diverse_hits_prefers_distinct_doc_ids() -> None:
    hits = [
        _doc(score=0.95, doc_id="a", chunk_index=0),
        _doc(score=0.9, doc_id="a", chunk_index=1),
        _doc(score=0.85, doc_id="b", chunk_index=0),
    ]

    selected = select_source_diverse_hits(hits, max_hits=2)

    assert [hit["doc_id"] for hit in selected] == ["a", "b"]


def test_doc_pipeline_comparison_keeps_source_diverse_blocks(
    doc_pipeline_io: dict,
) -> None:
    doc_pipeline_io["docs"] = [
        _doc(score=0.95, doc_id="a", chunk_index=0),
        _doc(score=0.9, doc_id="a", chunk_index=1),
        _doc(score=0.86, doc_id="b", chunk_index=0),
    ]

    result = retrieve_docs_for_rag(
        "A 和 B 有什么区别", query_type=QUERY_TYPE_COMPARISON
    )

    assert [hit["doc_id"] for hit in result.merged_doc_hits] == ["a", "b"]
    assert result.retrieval_debug["source_diversity_enabled"] is True
    assert result.retrieval_debug["source_diversity_doc_ids"] == ["a", "b"]


# ---------------------------------------------------------------------------
# memory_pipeline.filter_memory_hits （纯）
# ---------------------------------------------------------------------------


def test_filter_memory_hits_drops_summary_keywords() -> None:
    hits = [
        {"content": "用户请求总结最近对话"},
        {"content": "WAI-ARIA 是无障碍规范"},
        {"content": "帮我回顾一下刚才的问题"},
    ]
    filtered = filter_memory_hits(hits)
    # 两条 meta query 被过滤
    assert len(filtered) == 1
    assert "WAI-ARIA" in filtered[0]["content"]


def test_filter_memory_hits_drops_recall_keywords() -> None:
    hits = [{"content": "之前用户问过的天气"}]
    assert filter_memory_hits(hits) == []


def test_filter_memory_hits_empty_list() -> None:
    assert filter_memory_hits([]) == []


# ---------------------------------------------------------------------------
# memory_pipeline.retrieve_memory_for_rag
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_pipeline_io(monkeypatch: pytest.MonkeyPatch) -> dict:
    import app.agents.rag.memory_pipeline as mp_mod

    harness: dict = {
        "hits": [],
        "search_error": None,
        "rerank_fn": lambda query, hits, top_k: hits[:top_k],
        "search_calls": 0,
        "rerank_calls": 0,
    }

    def fake_search_memory(query: str, top_k: int = 5, session_id: str = "") -> list:
        harness["search_calls"] += 1
        if harness["search_error"] is not None:
            raise harness["search_error"]
        return list(harness["hits"])

    def fake_rerank(query: str, hits: list, top_k: int = 5) -> list:
        harness["rerank_calls"] += 1
        return harness["rerank_fn"](query, hits, top_k)

    monkeypatch.setattr(mp_mod, "search_memory", fake_search_memory)
    monkeypatch.setattr(mp_mod, "rerank", fake_rerank)
    return harness


def test_memory_pipeline_disabled_skips_all_io(memory_pipeline_io: dict) -> None:
    """enabled=False 时完全不查 Chroma / 不 rerank。"""
    result = retrieve_memory_for_rag("q", session_id="s1", enabled=False)

    assert result.memory_hits == []
    assert memory_pipeline_io["search_calls"] == 0
    assert memory_pipeline_io["rerank_calls"] == 0
    assert result.retrieval_debug["enabled"] is False
    assert result.retrieval_debug["skip_reason"]  # 非空 skip_reason


def test_memory_pipeline_search_error_recorded(memory_pipeline_io: dict) -> None:
    memory_pipeline_io["search_error"] = RuntimeError("memory store down")
    result = retrieve_memory_for_rag("q", session_id="s1", enabled=True)
    assert result.memory_hits == []
    assert len(result.errors) == 1


def test_memory_pipeline_filters_meta_and_dedupes(memory_pipeline_io: dict) -> None:
    """搜回来的 hits：meta keyword 被滤掉，重复 memory_key 被去重。"""
    memory_pipeline_io["hits"] = [
        {"id": "m1", "content": "WAI-ARIA 是无障碍规范", "memory_key": "k1"},
        {"id": "m2", "content": "帮我回顾对话", "memory_key": "k2"},  # 会被过滤
        {
            "id": "m3",
            "content": "WAI-ARIA 是无障碍规范",
            "memory_key": "k1",
        },  # 同 key 去重
    ]
    result = retrieve_memory_for_rag("q", session_id="s1", enabled=True)
    # 剩 1 条
    assert len(result.memory_hits) == 1
    assert result.memory_hits[0]["memory_key"] == "k1"


def test_memory_pipeline_rerank_error_falls_back(memory_pipeline_io: dict) -> None:
    memory_pipeline_io["hits"] = [
        {"id": f"m{i}", "content": f"事实 {i}", "memory_key": f"k{i}"} for i in range(3)
    ]

    def raise_error(*_args, **_kwargs):
        raise RuntimeError("rerank died")

    memory_pipeline_io["rerank_fn"] = raise_error

    result = retrieve_memory_for_rag("q", session_id="s1", enabled=True)
    # 回退为 memory_before_rerank[:memory_rerank_top_k=5]
    assert len(result.memory_hits) == 3
    assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# adapt_strategy_max_tokens — 动态 max_token 收紧
# ---------------------------------------------------------------------------


def test_adapt_strategy_max_tokens_unchanged_when_context_large() -> None:
    """实际 context 足够长时，max_tokens 不变。"""

    strategy = {"name": "default_short", "context_chars": 360, "max_tokens": 180}
    adapted = adapt_strategy_max_tokens(strategy, actual_context_chars=360)

    # 360 // 2 = 180 == base，不需要复制新 dict
    assert adapted is strategy
    assert adapted["max_tokens"] == 180


def test_adapt_strategy_max_tokens_reduced_for_short_context() -> None:
    """实际 context 短时，max_tokens 向下收紧。"""

    strategy = {"name": "default_short", "context_chars": 360, "max_tokens": 180}
    adapted = adapt_strategy_max_tokens(strategy, actual_context_chars=100)

    # 100 // 2 = 50 < 180，应收紧到 50
    assert adapted["max_tokens"] == 50
    assert adapted is not strategy  # 返回新 dict，不修改原策略


def test_adapt_strategy_max_tokens_floors_at_minimum() -> None:
    """极短 context 时 max_tokens 不低于 ANSWER_TOKENS_MIN。"""

    strategy = {"name": "default_short", "context_chars": 360, "max_tokens": 180}
    adapted = adapt_strategy_max_tokens(strategy, actual_context_chars=20)

    # 20 // 2 = 10 < ANSWER_TOKENS_MIN，应保底
    assert adapted["max_tokens"] == ANSWER_TOKENS_MIN


def test_adapt_strategy_max_tokens_respects_strategy_ceiling() -> None:
    """即使 context 很长，max_tokens 也不超过策略本身的上限。"""

    strategy = {"name": "definition_short", "context_chars": 280, "max_tokens": 100}
    adapted = adapt_strategy_max_tokens(strategy, actual_context_chars=500)

    # 500 // 2 = 250 > 100，应被 base=100 截断
    assert adapted["max_tokens"] == 100
    assert adapted is strategy  # 无变化时返回同一对象


def test_adapt_strategy_preserves_other_fields() -> None:
    """adapt 只改 max_tokens，其他字段原样保留。"""

    strategy = {
        "name": "comparison",
        "answer_style": "用对比方式回答",
        "context_chars": 360,
        "max_tokens": 180,
    }
    adapted = adapt_strategy_max_tokens(strategy, actual_context_chars=80)

    assert adapted["name"] == "comparison"
    assert adapted["answer_style"] == "用对比方式回答"
    assert adapted["context_chars"] == 360
    assert adapted["max_tokens"] == 40  # 80 // 2

"""RAG pipeline 单测：doc_pipeline + memory_pipeline + rewrite。

E2E 测试把这三个 pipeline 整个 stub 掉了，内部分支全没覆盖。这里把 I/O
（Chroma 查询、rerank LLM、rewrite LLM）打到 pipeline 模块命名空间上，专门
驱动内部分支。
"""

from __future__ import annotations

import pytest

from app.agents.rag.doc_pipeline import retrieve_docs_for_rag
from app.agents.rag.memory_pipeline import filter_memory_hits, retrieve_memory_for_rag
from app.agents.rag.constants import QUERY_TYPE_COMPARISON, QUERY_TYPE_FALLBACK
from app.agents.rag.rewrite import (
    get_user_messages,
    rewrite_rag_query,
    simple_rewrite,
)


# ---------------------------------------------------------------------------
# rewrite.simple_rewrite （纯函数）
# ---------------------------------------------------------------------------


def test_simple_rewrite_handles_na_xxx_ne_pattern() -> None:
    """ "那 北京 呢" → "北京气候怎么样？"（确定性兜底，省一次 LLM）。"""
    assert simple_rewrite("那北京呢") == "北京气候怎么样？"
    assert simple_rewrite("  那深圳呢  ") == "深圳气候怎么样？"


def test_simple_rewrite_short_city_query() -> None:
    """4 字以内 & 不带问号 → 补成天气问法。"""
    assert simple_rewrite("北京") == "北京气候怎么样？"
    assert simple_rewrite("上海") == "上海气候怎么样？"


def test_simple_rewrite_returns_none_for_long_query() -> None:
    """长文本不做轻量改写，留给 LLM 或默认兜底。"""
    assert simple_rewrite("请问上海今天天气怎么样") is None


def test_simple_rewrite_returns_none_when_question_mark_present() -> None:
    """已经带问号的短问题不改写。"""
    assert simple_rewrite("北京？") is None


def test_simple_rewrite_returns_none_when_only_na_ne() -> None:
    """纯 "那呢" → city 为空，走不到 city 分支，但会落入短句兜底。"""
    # 行为记录：此处不会返回 None，而是被当作短句补成 "那呢气候怎么样？"。
    # 这更多是"短句兜底优先级高"的副作用，记录当前行为，防止将来静默改动。
    assert simple_rewrite("那呢") == "那呢气候怎么样？"


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


def test_rewrite_rag_query_simple_branch_skips_llm(llm_stub) -> None:
    """simple_rewrite 命中时，不应发起任何 LLM 调用。"""
    result = rewrite_rag_query("那北京呢", messages=[], summary="")
    assert result.query == "北京气候怎么样？"
    assert result.errors == []
    assert result.timing_ms >= 0
    # simple_rewrite 分支不调 LLM
    assert llm_stub.calls == []


def test_rewrite_rag_query_na_prefix_triggers_llm(llm_stub) -> None:
    """不命中 simple_rewrite 的 "那xxx..." → 走 LLM 改写路径。"""
    llm_stub.set_response("北京今天气温多少度")

    result = rewrite_rag_query(
        "那首都现在情况如何",
        messages=[{"role": "user", "content": "上海气温"}],
        summary="用户关注各城市气温",
    )
    assert result.query == "北京今天气温多少度"
    assert result.errors == []
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

    result = rewrite_rag_query("那上海怎么样", messages=[], summary="")
    # 原 message 不以问号结尾 → 补上
    assert result.query.endswith("？")
    # 错误信息被记录
    assert len(result.errors) == 1


def test_rewrite_rag_query_default_branch_appends_question_mark(llm_stub) -> None:
    """既不命中 simple_rewrite、又不以"那"开头 → 默认直接补问号。"""
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
    result = retrieve_docs_for_rag("A 和 B 有什么区别", query_type=QUERY_TYPE_COMPARISON)

    assert result.retrieval_debug["query_type"] == QUERY_TYPE_COMPARISON
    assert result.retrieval_debug["requested_top_k"] >= 8
    assert result.retrieval_debug["candidate_top_k"] >= 32
    assert "dense_search" in result.retrieval_debug["pipeline_steps"]
    assert "lexical_search" in result.retrieval_debug["pipeline_steps"]
    assert "hybrid_merge" in result.retrieval_debug["pipeline_steps"]
    assert "threshold" in result.retrieval_debug["pipeline_steps"]


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

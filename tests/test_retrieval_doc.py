"""RAG 文档检索的纯计算函数测试。

这批测试**不触碰 Chroma / 不触碰 embedding / 不需要 llm_stub**：
目标函数都是纯计算（输入 → 输出），所以跑得快、断言直观。

覆盖的职责点：
- `distance_to_semantic_score` / `keyword_score` / `normalize_keyword_scores`：打分原子
- `merge_doc_hits`：多路召回去重 + 来源合并
- `rank_hybrid`：最终排序（hybrid 权重 + 平局拆分）

和 Chroma / embedding 绑定的 `search_docs` / `dense_retrieve_docs` / `keyword_retrieve_docs`
走 Week 2 晚些时候的集成测试补；本文件只保护核心排序逻辑。
"""

from __future__ import annotations

from app.retrieval.doc_retrieval import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_HYBRID_BETA,
    RETRIEVAL_SOURCE_DENSE,
    RETRIEVAL_SOURCE_KEYWORD,
    distance_to_semantic_score,
    keyword_score,
    merge_doc_hits,
    normalize_keyword_scores,
    rank_hybrid,
)

# ----------------------------- distance → semantic -----------------------------

def test_distance_to_semantic_score_basic() -> None:
    # cosine distance 0 表示完全一致 → semantic 1.0
    assert distance_to_semantic_score(0.0) == 1.0
    # distance 1 表示正交 → semantic 0.0
    assert distance_to_semantic_score(1.0) == 0.0
    # distance 略大于 1（Chroma 偶有浮点误差）应 clamp 到 0，不变负数
    assert distance_to_semantic_score(1.2) == 0.0
    # None（Chroma 返回里偶尔缺字段）按 0 处理，不应该抛
    assert distance_to_semantic_score(None) == 0.0


# ----------------------------- keyword_score -----------------------------

def test_keyword_score_tag_match_hits_weight() -> None:
    # query 里提取到 "virtual-list" 标签，content 里包含匹配词 "虚拟列表"
    # → 加上 CONCEPT_TAG_WEIGHTS[virtual-list] = 1.5
    # query "虚拟列表是什么" 不是 content "虚拟列表是一种..." 的子串，所以没有 LITERAL 加成
    score = keyword_score("虚拟列表是什么", "虚拟列表是一种前端性能优化技术")
    assert score == 1.5


def test_keyword_score_no_match_returns_zero() -> None:
    # 既没有标签命中，也没有字面子串命中
    assert keyword_score("XYZ", "完全无关的内容") == 0.0


def test_keyword_score_literal_substring_adds_boost() -> None:
    # 短 query 整个出现在 content 里 → 加 LITERAL_MATCH_WEIGHT (1.5)
    # 这里故意不触发任何 tag 匹配，隔离验证 literal 通路。
    score = keyword_score("abcd", "xxx abcd yyy")
    assert score == 1.5


# ----------------------------- normalize_keyword_scores -----------------------------

def test_normalize_keyword_scores_scales_to_unit_range() -> None:
    hits = [
        {"id": "a", "keyword_score": 1.0},
        {"id": "b", "keyword_score": 2.0},
        {"id": "c", "keyword_score": 4.0},  # max
    ]
    out = normalize_keyword_scores(hits)
    # max 归一到 1.0，其余按比例
    by_id = {h["id"]: h["keyword_score_norm"] for h in out}
    assert by_id["c"] == 1.0
    assert by_id["b"] == 0.5
    assert by_id["a"] == 0.25


def test_normalize_keyword_scores_all_zero_sets_norm_zero() -> None:
    # 所有命中 keyword 分都是 0（纯 dense 召回时常见）→ norm 应全为 0，不应除零
    hits = [{"id": "a", "keyword_score": 0.0}, {"id": "b", "keyword_score": 0.0}]
    out = normalize_keyword_scores(hits)
    assert all(h["keyword_score_norm"] == 0.0 for h in out)


def test_apply_keyword_scores_preserves_existing_lexical_score() -> None:
    from app.retrieval.doc_retrieval import apply_keyword_scores

    hits = [
        {
            "id": "fts-hit",
            "content": "完全不触发项目标签的内容",
            "keyword_score": 0.8,
        }
    ]

    out = apply_keyword_scores("unmatched-query", hits)

    assert out[0]["keyword_score"] == 0.8
    assert out[0]["keyword_score_norm"] == 1.0


# ----------------------------- merge_doc_hits -----------------------------

def test_merge_doc_hits_dedupes_and_merges_sources() -> None:
    # 同一个 chunk id 被 dense 和 keyword 两路都召回
    dense = [
        {
            "id": "chunk-1",
            "content": "text",
            "semantic_score": 0.8,
            "distance": 0.2,
            "retrieval_source": RETRIEVAL_SOURCE_DENSE,
        }
    ]
    keyword = [
        {
            "id": "chunk-1",
            "content": "text",
            "keyword_score": 3.0,
            "retrieval_source": RETRIEVAL_SOURCE_KEYWORD,
        }
    ]
    merged = merge_doc_hits([dense, keyword])

    assert len(merged) == 1
    hit = merged[0]
    # 两路来源都应当被记录下来（后续 debug_info 依赖这个字段）
    assert set(hit["retrieval_sources"]) == {
        RETRIEVAL_SOURCE_DENSE,
        RETRIEVAL_SOURCE_KEYWORD,
    }
    # semantic / keyword 各取更大值
    assert hit["semantic_score"] == 0.8
    assert hit["keyword_score"] == 3.0


def test_merge_doc_hits_skips_empty_ids() -> None:
    # 没有 id 的 hit 直接丢弃，避免后续 dedup 把它们糊在一起
    merged = merge_doc_hits([[{"id": "", "content": "x"}]])
    assert merged == []


# ----------------------------- rank_hybrid -----------------------------

def test_rank_hybrid_semantic_dominates_with_default_weights() -> None:
    # alpha=0.65 给 semantic，beta=0.35 给 keyword
    # 两条 hit 的 semantic 差距足够大时，不应被 keyword 翻盘
    hits = [
        {"id": "sem_high", "semantic_score": 0.95, "keyword_score_norm": 0.1},
        {"id": "kw_high", "semantic_score": 0.10, "keyword_score_norm": 0.95},
    ]
    ranked = rank_hybrid(
        hits, alpha=DEFAULT_HYBRID_ALPHA, beta=DEFAULT_HYBRID_BETA
    )
    # 0.95*0.65 + 0.1*0.35 = 0.6525 vs 0.1*0.65 + 0.95*0.35 = 0.3975
    assert ranked[0]["id"] == "sem_high"
    # 顺手验证 score 被写回到 hit 上，后续 debug 会读
    assert "score" in ranked[0]


def test_rank_hybrid_keyword_breaks_tie_via_higher_score() -> None:
    # semantic 相同的情况下，keyword 更高的那个 hybrid score 更高
    hits = [
        {"id": "low_kw", "semantic_score": 0.5, "keyword_score_norm": 0.2},
        {"id": "high_kw", "semantic_score": 0.5, "keyword_score_norm": 0.9},
    ]
    ranked = rank_hybrid(
        hits, alpha=DEFAULT_HYBRID_ALPHA, beta=DEFAULT_HYBRID_BETA
    )
    assert ranked[0]["id"] == "high_kw"


def test_rank_hybrid_tolerates_lexical_only_hits() -> None:
    # SQLite FTS lexical hit 没有 semantic_score 时，应按 0 处理，不能让 RAG 500。
    hits = [{"id": "lexical_only", "keyword_score_norm": 1.0}]

    ranked = rank_hybrid(hits)

    assert ranked[0]["id"] == "lexical_only"
    assert ranked[0]["semantic_score"] == 0.0
    assert ranked[0]["score"] == DEFAULT_HYBRID_BETA

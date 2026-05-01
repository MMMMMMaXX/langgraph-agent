from app.knowledge.search_inspector import inspect_retrieval


def _hit(
    chunk_id: str,
    *,
    doc_id: str = "doc1",
    content: str = "WAI-ARIA 是无障碍技术规范。",
    score: float = 0.8,
    keyword: float = 1.0,
) -> dict:
    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "doc_title": "测试文档",
        "source": "unit.md",
        "section_title": "Intro",
        "chunk_index": 0,
        "chunk_char_len": len(content),
        "content": content,
        "score": score,
        "semantic_score": score,
        "keyword_score_norm": keyword,
        "retrieval_sources": ["dense"],
    }


def test_inspect_retrieval_exposes_pipeline_steps(monkeypatch) -> None:
    import app.agents.rag.doc_pipeline as dp_mod

    dense_hit = _hit("doc1::chunk::0", score=0.8, keyword=0.0)
    lexical_hit = _hit("doc1::chunk::0", score=0.7, keyword=1.0)

    monkeypatch.setattr(dp_mod, "dense_retrieve_docs", lambda query, top_k: [dense_hit])
    monkeypatch.setattr(
        dp_mod,
        "keyword_retrieve_docs",
        lambda query, top_k: [lexical_hit],
    )

    report = inspect_retrieval("WAI-ARIA 是什么", top_k=3)

    assert report.query_type == "definition"
    assert report.retrieval_debug["dense_count"] == 1
    assert report.retrieval_debug["lexical_count"] == 1
    assert report.retrieval_debug["filtered_count"] == 1
    assert report.dense_hits[0]["id"] == "doc1::chunk::0"
    assert report.lexical_hits[0]["id"] == "doc1::chunk::0"
    assert report.hybrid_hits[0]["rank"] == 1
    assert report.hybrid_hits[0]["keyword_score_norm"] == 1.0
    assert "bm25_score_norm" in report.hybrid_hits[0]
    assert "lexical_content_score_norm" in report.hybrid_hits[0]
    assert report.filtered_hits
    assert report.reranked_hits
    assert report.merged_hits
    assert report.stage_metrics["counts"]["dense"] == 1
    assert report.stage_metrics["counts"]["lexical"] == 1
    assert report.stage_metrics["top_ids"]["hybrid"] == ["doc1::chunk::0"]
    assert report.stage_metrics["score_weights"]["hybrid_alpha"] > 0
    assert report.stage_metrics["context"]["citation_count"] == 1
    assert report.citations[0]["ref"] == "[1]"
    assert "WAI-ARIA" in report.context_preview


def test_inspect_retrieval_rejects_empty_query() -> None:
    try:
        inspect_retrieval("   ")
    except ValueError as exc:
        assert str(exc) == "query must not be empty"
    else:
        raise AssertionError("expected ValueError")

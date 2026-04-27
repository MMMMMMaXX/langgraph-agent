from app.retrieval.lexical.tokenizer import (
    build_fts_index_text,
    build_fts_query,
    lexical_terms,
)


def test_lexical_terms_include_ascii_jieba_and_bigrams() -> None:
    terms = lexical_terms("WAI-ARIA 虚拟列表是什么")

    assert "WAI-ARIA" in terms
    assert "虚拟列表" in terms or "虚拟" in terms
    assert "列表" in terms


def test_build_fts_query_removes_common_question_stopwords() -> None:
    query = build_fts_query("虚拟列表是什么")

    assert "是什么" not in query
    assert "列表" in query


def test_build_fts_index_text_preserves_original_text_and_terms() -> None:
    text = "虚拟列表是一种前端性能优化技术"

    index_text = build_fts_index_text(text)

    assert text in index_text
    assert "虚拟列表" in index_text or "虚拟" in index_text

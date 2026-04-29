from app.retrieval.reranker import (
    RERANK_CONTENT_MAX_CHARS,
    format_rerank_candidate,
    rerank,
)


def test_format_rerank_candidate_includes_metadata_scores_and_sources() -> None:
    text = format_rerank_candidate(
        {
            "doc_title": "RAG 设计",
            "source": "docs/rag.md",
            "section_title": "重排",
            "score": 0.81234,
            "semantic_score": 0.72345,
            "keyword_score_norm": 0.93456,
            "retrieval_sources": ["dense", "keyword"],
            "content": "chunk 正文",
        }
    )

    assert "标题: RAG 设计" in text
    assert "来源: docs/rag.md" in text
    assert "章节: 重排" in text
    assert "综合分: 0.8123" in text
    assert "语义分: 0.7235" in text
    assert "关键词分: 0.9346" in text
    assert "召回来源: dense, keyword" in text
    assert "正文: chunk 正文" in text


def test_format_rerank_candidate_truncates_only_abnormally_long_content() -> None:
    content = "我" * (RERANK_CONTENT_MAX_CHARS + 10)

    text = format_rerank_candidate({"content": content})

    assert f"正文: {'我' * RERANK_CONTENT_MAX_CHARS}..." == text
    assert len(text) == len("正文: ") + RERANK_CONTENT_MAX_CHARS + len("...")


def test_format_rerank_candidate_keeps_normal_chunk_content() -> None:
    content = "正常 chunk 内容"

    text = format_rerank_candidate({"content": content})

    assert text == "正文: 正常 chunk 内容"


def test_rerank_sends_formatted_candidates_to_prompt(monkeypatch) -> None:
    captured: dict[str, object] = {}
    candidates = [
        {
            "id": "a",
            "doc_title": "文档 A",
            "score": 0.9,
            "retrieval_sources": ["dense"],
            "content": "A 内容",
        },
        {
            "id": "b",
            "doc_title": "文档 B",
            "score": 0.7,
            "retrieval_sources": ["keyword"],
            "content": "B 内容",
        },
    ]

    def fake_build_rerank_prompt(query: str, texts: list[str]) -> str:
        captured["query"] = query
        captured["texts"] = texts
        return "prompt"

    def fake_chat(messages: list[dict]) -> str:
        captured["messages"] = messages
        return "[1,0]"

    monkeypatch.setattr(
        "app.retrieval.reranker.build_rerank_prompt",
        fake_build_rerank_prompt,
    )
    monkeypatch.setattr("app.retrieval.reranker.chat", fake_chat)

    result = rerank("怎么排序", candidates, top_k=2)

    assert result == [candidates[1], candidates[0]]
    assert captured["query"] == "怎么排序"
    assert captured["messages"] == [{"role": "user", "content": "prompt"}]
    texts = captured["texts"]
    assert isinstance(texts, list)
    assert "标题: 文档 A" in texts[0]
    assert "综合分: 0.9000" in texts[0]
    assert "召回来源: dense" in texts[0]
    assert "标题: 文档 B" in texts[1]
    assert "召回来源: keyword" in texts[1]

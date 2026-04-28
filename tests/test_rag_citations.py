from app.agents.rag.context import build_rag_context


def test_build_rag_context_adds_cited_doc_blocks() -> None:
    doc_hit = {
        "id": "0::chunk::1",
        "doc_id": "0",
        "doc_title": "WAI-ARIA 文档",
        "source": "db.json",
        "section_title": "无障碍",
        "chunk_index": 1,
        "content": "WAI-ARIA 是一个无障碍技术规范。",
        "score": 0.7558,
    }

    context = build_rag_context(
        doc_hits=[doc_hit],
        memory_hits=[],
        doc_context_chars=200,
    )

    assert "[1]" in context.doc_context
    assert "来源：db.json" in context.doc_context
    assert "WAI-ARIA 是一个无障碍技术规范。" in context.doc_context
    assert context.citations[0]["ref"] == "[1]"
    assert context.citations[0]["doc_id"] == "0"
    assert context.citations[0]["chunk_id"] == "0::chunk::1"


def test_build_rag_context_citation_uses_merged_chunk_ids() -> None:
    doc_hit = {
        "id": "0::chunk::1+0::chunk::2",
        "doc_id": "0",
        "merged_chunk_ids": ["0::chunk::1", "0::chunk::2"],
        "content": "合并后的文档内容",
    }

    context = build_rag_context(
        doc_hits=[doc_hit],
        memory_hits=[],
        doc_context_chars=200,
    )

    assert context.citations[0]["chunk_id"] == "0::chunk::1+0::chunk::2"
    assert context.citations[0]["merged_chunk_ids"] == [
        "0::chunk::1",
        "0::chunk::2",
    ]


def test_build_rag_context_skips_empty_doc_without_citation_mismatch() -> None:
    context = build_rag_context(
        doc_hits=[
            {"id": "empty", "doc_id": "empty", "content": ""},
            {"id": "real", "doc_id": "real", "content": "真实内容"},
        ],
        memory_hits=[],
        doc_context_chars=200,
    )

    assert "[1]" in context.doc_context
    assert "真实内容" in context.doc_context
    assert context.citations[0]["doc_id"] == "real"

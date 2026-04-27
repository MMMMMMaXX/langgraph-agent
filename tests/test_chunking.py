from app.chunking import chunk_document_text


def test_chunking_preserves_markdown_section_title() -> None:
    text = "# WAI-ARIA\n\nWAI-ARIA 是无障碍技术规范。\n\n它可以帮助屏幕阅读器识别状态。"

    chunks = chunk_document_text(
        "doc1",
        text,
        chunk_size_chars=120,
        chunk_overlap_chars=20,
        min_chunk_chars=10,
    )

    assert chunks
    assert chunks[0].section_title == "WAI-ARIA"
    assert "WAI-ARIA" in chunks[0].text


def test_chunking_long_paragraph_falls_back_to_window() -> None:
    long_text = "虚拟列表" + "性能优化" * 80

    chunks = chunk_document_text(
        "doc2",
        long_text,
        chunk_size_chars=80,
        chunk_overlap_chars=10,
        min_chunk_chars=20,
    )

    assert len(chunks) > 1
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1

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


def test_chunking_uses_smaller_windows_for_deep_markdown_sections() -> None:
    paragraph = "配置步骤" * 32
    body = f"{paragraph}\n\n{paragraph}"

    top_level_chunks = chunk_document_text(
        "doc3",
        f"# 指南\n\n{body}",
        source_type="md",
    )
    deep_level_chunks = chunk_document_text(
        "doc4",
        f"### 配置\n\n{body}",
        source_type="md",
    )

    assert len(top_level_chunks) == 1
    assert len(deep_level_chunks) > 1
    assert all(chunk.section_title == "配置" for chunk in deep_level_chunks)


def test_chunking_uses_smaller_windows_for_json_documents() -> None:
    content = "结构化字段" * 55

    chunks = chunk_document_text("doc5", content, source_type="json")

    assert len(chunks) > 1
    assert max(chunk.char_len for chunk in chunks) <= 240


def test_chunking_explicit_size_overrides_dynamic_source_type() -> None:
    content = "结构化字段" * 45

    chunks = chunk_document_text(
        "doc6",
        content,
        source_type="json",
        chunk_size_chars=320,
        chunk_overlap_chars=20,
        min_chunk_chars=20,
    )

    assert len(chunks) == 1

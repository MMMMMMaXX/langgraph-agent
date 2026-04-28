from app.agents.rag.constants import QUERY_TYPE_COMPARISON, QUERY_TYPE_DEFINITION
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


def test_build_rag_context_compresses_irrelevant_sentences_but_keeps_citation() -> None:
    doc_hit = {
        "id": "3::chunk::0",
        "doc_id": "3",
        "doc_title": "虚拟列表",
        "source": "db.json",
        "chunk_index": 0,
        "content": (
            "这是一句和主题无关的背景说明。"
            "虚拟列表只对可见区域进行渲染，对非可见区域不渲染。"
            "它能提升大量数据列表的渲染性能。"
            "这一句继续补充无关背景。"
        ),
    }

    context = build_rag_context(
        doc_hits=[doc_hit],
        memory_hits=[],
        doc_context_chars=80,
        query="虚拟列表是什么？",
        query_type=QUERY_TYPE_DEFINITION,
    )

    assert "[1]" in context.doc_context
    assert "虚拟列表只对可见区域进行渲染" in context.doc_context
    assert context.context_compression["before_chars"] > context.context_compression[
        "after_chars"
    ]
    assert context.context_compression["compression_ratio"] < 1
    assert context.citations[0]["doc_id"] == "3"


def test_build_rag_context_keeps_multiple_sources_for_comparison() -> None:
    wai_aria_hit = {
        "id": "0::chunk::1",
        "doc_id": "0",
        "doc_title": "WAI-ARIA",
        "source": "db.json",
        "chunk_index": 1,
        "content": "WAI-ARIA 是无障碍技术规范。它让屏幕阅读器识别页面状态。",
    }
    virtual_list_hit = {
        "id": "3::chunk::0",
        "doc_id": "3",
        "doc_title": "虚拟列表",
        "source": "db.json",
        "chunk_index": 0,
        "content": "虚拟列表只渲染可见区域。它主要用于提升大量数据渲染性能。",
    }

    context = build_rag_context(
        doc_hits=[wai_aria_hit, virtual_list_hit],
        memory_hits=[],
        doc_context_chars=120,
        query="WAI-ARIA 和虚拟列表有什么区别？",
        query_type=QUERY_TYPE_COMPARISON,
    )

    assert "[1]" in context.doc_context
    assert "[2]" in context.doc_context
    assert "WAI-ARIA 是无障碍技术规范" in context.doc_context
    assert "虚拟列表只渲染可见区域" in context.doc_context
    assert [citation["doc_id"] for citation in context.citations] == ["0", "3"]

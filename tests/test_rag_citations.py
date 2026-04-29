from app.agents.rag.constants import QUERY_TYPE_COMPARISON, QUERY_TYPE_DEFINITION
from app.agents.rag.context import build_rag_context, compress_memory_context
from app.agents.rag.answer import check_citation_coverage


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
    assert (
        context.context_compression["before_chars"]
        > context.context_compression["after_chars"]
    )
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


# ===== compress_memory_context 测试 =====


def test_compress_memory_context_empty_hits_returns_empty() -> None:
    text, stats = compress_memory_context([])

    assert text == ""
    assert stats["hits_used"] == 0
    assert stats["hits_available"] == 0
    assert stats["before_chars"] == 0
    assert stats["compression_ratio"] == 0.0


def test_compress_memory_context_compresses_long_hit() -> None:
    """单条 memory 内容超过 block 上限时应被截断压缩。"""

    long_content = (
        "虚拟列表只渲染可见区域，不渲染屏幕外的元素。"
        "这是一条和当前问题无关的说明。"
        "它能大幅提升大量数据场景下的渲染性能。"
        "还有更多不相关的背景信息补充在这里。"
        "以及另一句无关的说明文字，继续撑大内容体积。"
    )
    hits = [{"content": long_content, "score": 0.9}]

    text, stats = compress_memory_context(
        hits,
        query="虚拟列表是什么",
        query_type=QUERY_TYPE_DEFINITION,
    )

    assert len(text) <= 120  # MEMORY_COMPRESSION_MAX_BLOCK_CHARS
    assert stats["hits_used"] == 1
    assert stats["before_chars"] > stats["after_chars"]
    assert stats["compression_ratio"] < 1.0


def test_compress_memory_context_respects_total_budget() -> None:
    """多条 memory 命中总量不超过 MEMORY_COMPRESSION_MAX_TOTAL_CHARS。"""

    hits = [
        {"content": "A" * 200, "score": 0.9},
        {"content": "B" * 200, "score": 0.8},
        {"content": "C" * 200, "score": 0.7},
        {"content": "D" * 200, "score": 0.6},  # 第 4 条应被丢弃（超出 max_hits=3）
    ]

    text, stats = compress_memory_context(hits)

    # stats["after_chars"] 只计内容字符不含 join 换行符，它是预算的实际约束对象
    assert stats["after_chars"] <= 360  # MEMORY_COMPRESSION_MAX_TOTAL_CHARS
    assert stats["hits_available"] == 4
    assert stats["hits_used"] <= 3  # MEMORY_COMPRESSION_MAX_HITS


def test_compress_memory_context_preserves_query_relevant_sentences() -> None:
    """包含 query 关键词的句子应被优先保留。"""

    content = (
        "这是一段无关背景说明，不含关键词。"
        "LangGraph 是一个多智能体编排框架。"
        "另一句完全无关的内容。"
    )
    hits = [{"content": content, "score": 0.9}]

    text, stats = compress_memory_context(
        hits,
        query="LangGraph 是什么",
        query_type=QUERY_TYPE_DEFINITION,
    )

    assert "LangGraph" in text


def test_build_rag_context_memory_is_compressed() -> None:
    """build_rag_context 中 memory_hits 的内容应经过压缩，不再是原始全量拼接。"""

    long_memory = (
        "北京气候属于温带季风气候，四季分明。"
        "这是一句和当前问题无关的补充说明。"
        "夏季炎热多雨，冬季寒冷干燥，春秋较短。"
        "再补充更多无关背景来撑大原始内容体积。"
        "最后一句继续添加冗余内容。"
    )
    memory_hits = [{"content": long_memory, "score": 0.9}]

    context = build_rag_context(
        doc_hits=[],
        memory_hits=memory_hits,
        doc_context_chars=200,
        query="北京气候怎么样",
        query_type=QUERY_TYPE_DEFINITION,
    )

    assert context.memory_compression["enabled"] is True
    assert context.memory_compression["hits_used"] == 1
    assert context.memory_compression["before_chars"] == len(long_memory)
    # 压缩后应短于原始内容
    assert (
        context.memory_compression["after_chars"]
        < context.memory_compression["before_chars"]
    )
    # memory_context 和 context 应使用同一份压缩结果（不再有 score 过滤不一致的 bug）
    assert context.memory_context in context.context


def test_build_rag_context_memory_compression_stats_when_no_memory() -> None:
    """无 memory_hits 时 memory_compression 应返回空统计，不报错。"""

    context = build_rag_context(
        doc_hits=[],
        memory_hits=[],
        doc_context_chars=200,
    )

    assert context.memory_compression["hits_used"] == 0
    assert context.memory_compression["hits_available"] == 0
    assert context.memory_compression["before_chars"] == 0


# ===== check_citation_coverage 测试 =====


def _make_citations(*refs: str) -> list[dict]:
    return [{"ref": ref} for ref in refs]


def test_check_citation_coverage_true_when_ref_present() -> None:
    citations = _make_citations("[1]", "[2]")
    assert (
        check_citation_coverage(
            "WAI-ARIA 是无障碍规范 [1]，虚拟列表用于性能优化。", citations
        )
        is True
    )


def test_check_citation_coverage_true_when_partial_refs_present() -> None:
    """只要有一个 ref 出现即视为覆盖，不要求全部。"""

    citations = _make_citations("[1]", "[2]")
    assert check_citation_coverage("答案只引用了第二个来源 [2]。", citations) is True


def test_check_citation_coverage_false_when_no_ref_present() -> None:
    citations = _make_citations("[1]", "[2]")
    assert check_citation_coverage("答案里没有任何引用标记。", citations) is False


def test_check_citation_coverage_true_when_citations_empty() -> None:
    """没有 citations 时（无文档命中），直接返回 True，不应误报。"""

    assert check_citation_coverage("任何答案文本", []) is True


def test_check_citation_coverage_false_on_bare_number_without_brackets() -> None:
    """纯数字不算引用，必须是 [N] 格式。"""

    citations = _make_citations("[1]")
    assert (
        check_citation_coverage("答案提到了第1个来源，但没有括号格式。", citations)
        is False
    )

from app.knowledge import KnowledgeCatalog, KnowledgeChunkRecord


def test_knowledge_catalog_fts_search_returns_chunk(tmp_path) -> None:
    db_path = tmp_path / "knowledge.sqlite3"
    catalog = KnowledgeCatalog(db_path)
    catalog.reset()
    catalog.upsert_document(
        doc_id="doc1",
        title="网络无障碍",
        source="unit-test.md",
        content="WAI-ARIA 是无障碍技术规范。",
    )
    catalog.replace_chunks(
        [
            KnowledgeChunkRecord(
                chunk_id="doc1::chunk::0",
                doc_id="doc1",
                doc_title="网络无障碍",
                source="unit-test.md",
                section_title="WAI-ARIA",
                chunk_index=0,
                content="WAI-ARIA 是无障碍技术规范，可以帮助屏幕阅读器识别状态。",
                start_char=0,
                end_char=40,
                chunk_char_len=40,
            )
        ]
    )

    hits = catalog.search_chunks("WAI-ARIA 是什么", top_k=3)

    assert len(hits) == 1
    assert hits[0]["id"] == "doc1::chunk::0"
    assert hits[0]["keyword_score_norm"] == 1.0
    assert hits[0]["section_title"] == "WAI-ARIA"


def test_knowledge_catalog_fts_search_supports_chinese_bigram(tmp_path) -> None:
    db_path = tmp_path / "knowledge.sqlite3"
    catalog = KnowledgeCatalog(db_path)
    catalog.reset()
    catalog.upsert_document(
        doc_id="doc2",
        title="虚拟列表",
        source="unit-test.md",
        content="虚拟列表是一种前端性能优化技术。",
    )
    catalog.replace_chunks(
        [
            KnowledgeChunkRecord(
                chunk_id="doc2::chunk::0",
                doc_id="doc2",
                doc_title="虚拟列表",
                source="unit-test.md",
                chunk_index=0,
                content="虚拟列表是一种只渲染可见区域的数据渲染技术。",
                start_char=0,
                end_char=30,
                chunk_char_len=30,
            )
        ]
    )

    hits = catalog.search_chunks("虚拟列表是什么", top_k=3)

    assert hits
    assert hits[0]["id"] == "doc2::chunk::0"

from scripts.eval_chat import (
    build_retrieval_eval,
    load_cases,
    resolve_expected_doc_ids,
    setup_knowledge_imports,
)


def test_build_retrieval_eval_marks_stage_hits() -> None:
    case = {"expected_doc_ids": ["0"]}
    debug_nodes = {
        "rag_agent": {
            "doc_used": True,
            "top_docs": [{"doc_id": "0"}],
            "filtered_docs": [{"doc_id": "0"}],
            "post_rerank_docs": [{"doc_id": "0"}],
            "merged_docs": [{"doc_id": "0"}],
            "citations": [
                {"index": 1, "ref": "[1]", "doc_id": "0", "chunk_id": "0::chunk::1"}
            ],
            "retrieval_debug": {
                "doc": {
                    "dense_count": 2,
                    "lexical_count": 1,
                    "hybrid_count": 2,
                    "filtered_count": 1,
                    "consumed_count": 1,
                    "merged_count": 1,
                }
            },
        }
    }

    metrics = build_retrieval_eval(case, debug_nodes, "WAI-ARIA 是规范[1]")

    assert metrics["top_k_hit"] == "true"
    assert metrics["filtered_hit"] == "true"
    assert metrics["rerank_hit"] == "true"
    assert metrics["merged_hit"] == "true"
    assert metrics["citation_count"] == 1
    assert metrics["citation_hit"] == "true"
    assert metrics["citation_expected_doc_coverage"] == "1/1"
    assert metrics["citation_all_expected_docs_hit"] == "true"
    assert metrics["answer_citation_refs"] == "1"
    assert metrics["answer_has_citation"] == "true"
    assert metrics["citation_refs_valid"] == "true"
    assert metrics["dense_count"] == 2
    assert metrics["retrieval_failure_stage"] == ""


def test_build_retrieval_eval_reports_threshold_miss() -> None:
    case = {"expected_doc_ids": ["0"]}
    debug_nodes = {
        "rag_agent": {
            "top_docs": [{"doc_id": "0"}],
            "filtered_docs": [{"doc_id": "1"}],
            "post_rerank_docs": [],
            "merged_docs": [],
            "retrieval_debug": {"doc": {}},
        }
    }

    metrics = build_retrieval_eval(case, debug_nodes)

    assert metrics["top_k_hit"] == "true"
    assert metrics["filtered_hit"] == "false"
    assert metrics["retrieval_failure_stage"] == "threshold_miss"


def test_build_retrieval_eval_skips_cases_without_expected_ids() -> None:
    metrics = build_retrieval_eval({}, {"rag_agent": {}})

    assert metrics["top_k_hit"] == "-"
    assert metrics["retrieval_failure_stage"] == "-"


def test_build_retrieval_eval_reports_invalid_answer_citation() -> None:
    debug_nodes = {
        "rag_agent": {
            "doc_used": True,
            "citations": [{"index": 1, "doc_id": "0"}],
            "retrieval_debug": {"doc": {}},
        }
    }

    metrics = build_retrieval_eval({}, debug_nodes, "答案错误引用了不存在的来源[2]")

    assert metrics["answer_citation_refs"] == "2"
    assert metrics["answer_has_citation"] == "true"
    assert metrics["citation_refs_valid"] == "false"
    assert metrics["invalid_citation_refs"] == "2"
    assert metrics["unused_citation_refs"] == "1"


def test_build_retrieval_eval_reports_missing_answer_citation() -> None:
    debug_nodes = {
        "rag_agent": {
            "doc_used": True,
            "citations": [{"index": 1, "doc_id": "0"}],
            "retrieval_debug": {"doc": {}},
        }
    }

    metrics = build_retrieval_eval({}, debug_nodes, "答案没有引用编号")

    assert metrics["answer_citation_refs"] == "-"
    assert metrics["answer_has_citation"] == "false"
    assert metrics["citation_refs_valid"] == "true"
    assert metrics["unused_citation_refs"] == "1"


def test_build_retrieval_eval_reports_partial_expected_doc_coverage() -> None:
    case = {"expected_doc_ids": ["0", "3"]}
    debug_nodes = {
        "rag_agent": {
            "doc_used": True,
            "citations": [{"index": 1, "doc_id": "0"}],
            "retrieval_debug": {"doc": {}},
        }
    }

    metrics = build_retrieval_eval(case, debug_nodes, "只引用了一个来源[1]")

    assert metrics["citation_expected_doc_coverage"] == "1/2"
    assert metrics["citation_all_expected_docs_hit"] == "false"


def test_resolve_expected_doc_ids_adds_import_alias_doc_ids() -> None:
    case = {
        "expected_doc_ids": ["0"],
        "expected_import_aliases": ["skill_doc"],
    }

    resolved = resolve_expected_doc_ids(case, {"skill_doc": "doc-imported"})

    assert resolved["expected_doc_ids"] == ["0", "doc-imported"]
    assert case["expected_doc_ids"] == ["0"]


def test_eval_cases_include_skills_real_doc_questions() -> None:
    case_ids = {case["id"] for case in load_cases()}

    assert {
        "skills_definition_real_doc",
        "skill_md_authoring_real_doc",
        "progressive_disclosure_definition_real_doc",
        "when_to_use_skill_real_doc",
    }.issubset(case_ids)


def test_setup_knowledge_imports_returns_alias_to_doc_id() -> None:
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"doc_id": "doc-imported"}

    class FakeClient:
        def __init__(self) -> None:
            self.posts = []

        def post(self, path: str, json: dict):
            self.posts.append({"path": path, "json": json})
            return FakeResponse()

    client = FakeClient()
    case = {
        "setup_knowledge_imports": [
            {
                "alias": "skill_doc",
                "title": "Skill 文档",
                "content": "Skill 是能力模块。",
            }
        ]
    }

    alias_to_doc_id = setup_knowledge_imports(client, case)

    assert alias_to_doc_id == {"skill_doc": "doc-imported"}
    assert client.posts[0]["path"] == "/knowledge/import"
    assert client.posts[0]["json"]["title"] == "Skill 文档"


def test_build_retrieval_eval_reports_rerank_miss() -> None:
    case = {"expected_doc_ids": ["0"]}
    debug_nodes = {
        "rag_agent": {
            "top_docs": [{"doc_id": "0"}],
            "filtered_docs": [{"doc_id": "0"}],
            "post_rerank_docs": [{"doc_id": "1"}],  # 正确文档被 rerank 排出
            "merged_docs": [{"doc_id": "1"}],
            "retrieval_debug": {"doc": {}},
        }
    }

    metrics = build_retrieval_eval(case, debug_nodes)

    assert metrics["top_k_hit"] == "true"
    assert metrics["filtered_hit"] == "true"
    assert metrics["rerank_hit"] == "false"
    assert metrics["retrieval_failure_stage"] == "rerank_miss"


def test_build_retrieval_eval_reports_chunk_merge_miss() -> None:
    case = {"expected_doc_ids": ["0"]}
    debug_nodes = {
        "rag_agent": {
            "top_docs": [{"doc_id": "0"}],
            "filtered_docs": [{"doc_id": "0"}],
            "post_rerank_docs": [{"doc_id": "0"}],
            "merged_docs": [{"doc_id": "1"}],  # 合并阶段覆盖了正确文档
            "retrieval_debug": {"doc": {}},
        }
    }

    metrics = build_retrieval_eval(case, debug_nodes)

    assert metrics["top_k_hit"] == "true"
    assert metrics["filtered_hit"] == "true"
    assert metrics["rerank_hit"] == "true"
    assert metrics["merged_hit"] == "false"
    assert metrics["retrieval_failure_stage"] == "chunk_merge_miss"

from scripts.eval_chat import (
    build_knowledge_import_payload,
    build_manifest_payload,
    build_retrieval_eval,
    cleanup_chroma_dir,
    decide_chroma_keep_reason,
    determine_run_success,
    load_cases,
    manifest_path_for_output,
    resolve_expected_doc_ids,
    setup_knowledge_imports,
)


def test_build_retrieval_eval_marks_stage_hits() -> None:
    case = {"expected_doc_ids": ["0"]}
    debug_nodes = {
        "rag_agent": {
            "doc_used": True,
            "top_docs": [{"id": "0::chunk::1", "doc_id": "0"}],
            "filtered_docs": [{"id": "0::chunk::1", "doc_id": "0"}],
            "post_rerank_docs": [{"id": "0::chunk::1", "doc_id": "0"}],
            "merged_docs": [{"id": "0::chunk::1", "doc_id": "0"}],
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
    assert metrics["source_doc_ids"] == "0"
    assert metrics["used_chunk_ids"] == "0::chunk::1"
    assert metrics["top_k_doc_ids"] == "0"
    assert metrics["top_k_chunk_ids"] == "0::chunk::1"
    assert metrics["citation_hit"] == "true"
    assert metrics["citation_expected_doc_coverage"] == "1/1"
    assert metrics["citation_all_expected_docs_hit"] == "true"
    assert metrics["answer_citation_refs"] == "1"
    assert metrics["answer_has_citation"] == "true"
    assert metrics["citation_refs_valid"] == "true"
    assert metrics["dense_count"] == 2
    assert metrics["retrieval_failure_stage"] == ""


def test_build_retrieval_eval_reports_fallback_accuracy() -> None:
    case = {"category": "fallback", "expected_fallback": True}

    metrics = build_retrieval_eval(case, {"rag_agent": {}}, "资料不足")

    assert metrics["fallback_accuracy"] == "true"


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


def test_build_retrieval_eval_prefers_expected_chunk_over_doc_match() -> None:
    case = {
        "expected_doc_ids": ["doc-1"],
        "expected_chunk_ids": ["doc-1::chunk::2"],
    }
    debug_nodes = {
        "rag_agent": {
            "top_docs": [{"doc_id": "doc-1", "id": "doc-1::chunk::4"}],
            "filtered_docs": [{"doc_id": "doc-1", "id": "doc-1::chunk::2"}],
            "post_rerank_docs": [{"doc_id": "doc-1", "id": "doc-1::chunk::4"}],
            "merged_docs": [{"doc_id": "doc-1", "id": "doc-1::chunk::4"}],
            "citations": [{"doc_id": "doc-1", "chunk_id": "doc-1::chunk::4"}],
            "retrieval_debug": {"doc": {}},
        }
    }

    metrics = build_retrieval_eval(case, debug_nodes, "引用了同文档的错误段落[1]")

    assert metrics["top_k_hit"] == "false"
    assert metrics["filtered_hit"] == "true"
    assert metrics["rerank_hit"] == "false"
    assert metrics["citation_hit"] == "false"
    assert metrics["citation_all_expected_docs_hit"] == "true"


def test_resolve_expected_doc_ids_adds_import_alias_doc_ids() -> None:
    case = {
        "expected_doc_ids": ["0"],
        "expected_import_aliases": ["skill_doc"],
    }

    resolved = resolve_expected_doc_ids(case, {"skill_doc": "doc-imported"})

    assert resolved["expected_doc_ids"] == ["0", "doc-imported"]
    assert case["expected_doc_ids"] == ["0"]


def test_resolve_expected_doc_ids_adds_import_alias_chunk_ids() -> None:
    case = {
        "expected_chunk_ids": ["0::chunk::1"],
        "expected_import_chunks": [{"alias": "skill_doc", "chunk_index": 2}],
    }

    resolved = resolve_expected_doc_ids(case, {"skill_doc": "doc-imported"})

    assert resolved["expected_chunk_ids"] == [
        "0::chunk::1",
        "doc-imported::chunk::2",
    ]
    assert case["expected_chunk_ids"] == ["0::chunk::1"]


def test_eval_cases_include_skills_real_doc_questions() -> None:
    case_ids = {case["id"] for case in load_cases()}

    assert {
        "skills_definition_real_doc",
        "skill_md_authoring_real_doc",
        "progressive_disclosure_definition_real_doc",
        "when_to_use_skill_real_doc",
        "skill_script_usage_real_doc",
        "skill_success_criteria_real_doc",
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
    assert "alias" not in client.posts[0]["json"]


def test_build_knowledge_import_payload_reads_content_path() -> None:
    payload = build_knowledge_import_payload(
        {
            "alias": "skills_doc",
            "title": "Skills 文档",
            "source_type": "md",
            "content_path": "scripts/eval_fixtures/how_to_use_skills.md",
        }
    )

    assert payload["title"] == "Skills 文档"
    assert payload["source_type"] == "md"
    assert "Agent Skills" in payload["content"]
    assert "alias" not in payload
    assert "content_path" not in payload


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


def test_decide_chroma_keep_reason_priority() -> None:
    # 外部目录优先级最高：即使 run 成功也不能动用户/包装器自己管理的目录。
    assert (
        decide_chroma_keep_reason(
            auto_created=False, keep_flag=False, run_succeeded=True
        )
        == "external_persist_dir"
    )
    # 强制保留 flag 高于失败保留。
    assert (
        decide_chroma_keep_reason(
            auto_created=True, keep_flag=True, run_succeeded=False
        )
        == "keep_flag"
    )
    # 自动创建 + 失败 -> 保留供排查。
    assert (
        decide_chroma_keep_reason(
            auto_created=True, keep_flag=False, run_succeeded=False
        )
        == "failure"
    )
    # 自动创建 + 成功 + 无强制保留 -> 可清理（None）。
    assert (
        decide_chroma_keep_reason(
            auto_created=True, keep_flag=False, run_succeeded=True
        )
        is None
    )


def test_determine_run_success_handles_empty_and_mixed() -> None:
    success, failed, rate = determine_run_success([])
    assert success is True
    assert failed == 0
    assert rate == 0.0

    # 真实 case 结果使用 "assertion" 字段（参见 summarize_results / write_csv）。
    # 这里必须用同一字段，否则等价于“status 不存在 -> 全部失败”，不能反映成功路径。
    success, failed, rate = determine_run_success(
        [{"assertion": "pass"}, {"assertion": "pass"}]
    )
    assert success is True
    assert failed == 0
    assert rate == 1.0

    success, failed, rate = determine_run_success(
        [
            {"assertion": "pass"},
            {"assertion": "fail"},
            {"assertion": "error"},
        ]
    )
    assert success is False
    assert failed == 2
    assert abs(rate - 1 / 3) < 1e-6

    # 缺失 "assertion" 字段也算失败，避免结构变更后又退化成“全 pass”假象。
    success, failed, _ = determine_run_success([{"status": "pass"}])
    assert success is False
    assert failed == 1


def test_cleanup_chroma_dir_removes_directory(tmp_path) -> None:
    target = tmp_path / "chroma"
    target.mkdir()
    (target / "data_level0.bin").write_bytes(b"x" * 16)

    assert cleanup_chroma_dir(target) is True
    assert not target.exists()


def test_cleanup_chroma_dir_returns_false_when_missing(tmp_path) -> None:
    assert cleanup_chroma_dir(tmp_path / "nonexistent") is False


def test_manifest_path_uses_eval_manifest_suffix(tmp_path) -> None:
    json_path = tmp_path / "run.json"
    assert manifest_path_for_output(json_path) == tmp_path / "run.manifest.json"


def test_build_manifest_payload_round_trip() -> None:
    payload = build_manifest_payload(
        chroma_dir="/tmp/eval-chroma-x",
        chroma_auto_created=True,
        chroma_cleaned=False,
        chroma_keep_reason="failure",
        chroma_size_bytes=2048,
        run_status="failure",
        pass_rate=0.5,
        total=4,
        failed=2,
    )
    assert payload["chroma_persist_dir"] == "/tmp/eval-chroma-x"
    assert payload["chroma_auto_created"] is True
    assert payload["chroma_cleaned"] is False
    assert payload["chroma_keep_reason"] == "failure"
    assert payload["chroma_size_bytes"] == 2048
    assert payload["run_status"] == "failure"
    assert payload["pass_rate"] == 0.5
    assert payload["total_cases"] == 4
    assert payload["failed_cases"] == 2

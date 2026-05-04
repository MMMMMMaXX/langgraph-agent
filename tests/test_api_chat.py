"""HTTP API 层（/chat, /chat/stream）单测。

策略：把 chat_runner 里的 `run_chat_turn` 打桩成确定返回值，**只测**：
- routes / schemas / chat_runner 的 "验证 → 调度 → 组装响应" 逻辑
- session_store 的快照 / 写回是否被触发
- streaming.py 的 SSE 事件协议（start / chunk / done / error）

真正的 graph 执行由 test_e2e_graph.py 负责；这里不跨这个边界，保持测试快、稳、聚焦。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import app, clear_session_store, session_store
from app.knowledge.chunk_inspector import ChunkQualityReport
from app.knowledge.ingestion import KnowledgeImportResult
from app.knowledge.management import KnowledgeDeleteResult, KnowledgeReindexResult
from app.knowledge.rechunk_preview import RechunkPreviewReport
from app.knowledge.search_inspector import SearchInspectReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_sessions() -> Iterator[None]:
    """每个用例前后都清一次内存 session，避免互相污染。"""
    clear_session_store()
    yield
    clear_session_store()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def patch_run_chat_turn(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """拦截 chat_runner 里引用的 run_chat_turn。

    - `calls`：记录每次调用的 (state, message)
    - `result`：下一次调用会返回的 dict；测试可以直接改
    - `error`：若非 None，调用时抛出该异常（用于测试异常分支）
    """
    import app.api.chat_runner as runner_mod

    harness: dict[str, Any] = {
        "calls": [],
        "error": None,
        "result": {
            "answer": "stub answer",
            "routes": ["chat_agent"],
            "summary": "",
            "messages": [
                {"role": "user", "content": "__will_be_overwritten__"},
                {"role": "assistant", "content": "stub answer"},
            ],
            "node_timings": {"supervisor": 0.1, "chat_agent": 0.2},
            "debug_info": {"supervisor": {"route_reason": "rule"}},
            "streamed_answer": False,
        },
    }

    def fake_run_chat_turn(state: dict, message: str) -> dict:
        harness["calls"].append({"state": dict(state), "message": message})
        if harness["error"] is not None:
            raise harness["error"]
        # 让返回的 messages 反映真实用户输入，更像真实 run_chat_turn 行为
        result = dict(harness["result"])
        result["messages"] = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["answer"]},
        ]
        return result

    monkeypatch.setattr(runner_mod, "run_chat_turn", fake_run_chat_turn)
    return harness


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------


def test_chat_happy_path_returns_answer_and_routes(
    client: TestClient, patch_run_chat_turn: dict
) -> None:
    resp = client.post(
        "/chat",
        json={"session_id": "u1", "message": "你好"},
    )
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["session_id"] == "u1"
    assert body["answer"] == "stub answer"
    assert body["routes"] == ["chat_agent"]
    assert body["summary"] == ""
    # 非 debug 请求不返回 debug payload
    assert body.get("debug") is None
    # request_id 是 12 位 hex
    assert re.fullmatch(r"[0-9a-f]{12}", body["request_id"])

    # run_chat_turn 被调用一次，且拿到了去空格后的 session_id
    assert len(patch_run_chat_turn["calls"]) == 1
    call = patch_run_chat_turn["calls"][0]
    assert call["message"] == "你好"
    assert call["state"]["session_id"] == "u1"
    assert call["state"]["request_id"] == body["request_id"]


def test_chat_trims_session_id_whitespace(
    client: TestClient, patch_run_chat_turn: dict
) -> None:
    resp = client.post(
        "/chat",
        json={"session_id": "  u1  ", "message": "hi"},
    )
    assert resp.status_code == 200
    # 返回的 session_id 已 trim
    assert resp.json()["session_id"] == "u1"
    # run_chat_turn 收到的也是 trim 后的
    assert patch_run_chat_turn["calls"][0]["state"]["session_id"] == "u1"


def test_chat_debug_true_returns_debug_payload(
    client: TestClient, patch_run_chat_turn: dict
) -> None:
    resp = client.post(
        "/chat",
        json={"session_id": "u1", "message": "hi", "debug": True},
    )
    assert resp.status_code == 200
    body = resp.json()

    debug = body["debug"]
    assert debug is not None
    assert debug["node_timings"] == {"supervisor": 0.1, "chat_agent": 0.2}
    assert debug["nodes"] == {"supervisor": {"route_reason": "rule"}}
    # tracing 字段由 get_langsmith_runtime_info 填充，key 存在即可
    assert "langsmith" in debug["tracing"]


def test_chat_empty_session_id_returns_400(
    client: TestClient, patch_run_chat_turn: dict
) -> None:
    resp = client.post(
        "/chat",
        json={"session_id": "   ", "message": "hi"},
    )
    assert resp.status_code == 400
    assert "session_id" in resp.json()["detail"]
    # 提前报错，run_chat_turn 不应被调用
    assert patch_run_chat_turn["calls"] == []


def test_chat_value_error_mapped_to_400(
    client: TestClient, patch_run_chat_turn: dict
) -> None:
    """run_chat_turn 抛 ValueError（例如空 message）时应翻译成 400。"""
    patch_run_chat_turn["error"] = ValueError("message must not be empty")

    resp = client.post(
        "/chat",
        json={"session_id": "u1", "message": "   "},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "message must not be empty"


def test_chat_unexpected_exception_mapped_to_500(
    client: TestClient, patch_run_chat_turn: dict
) -> None:
    """任何未预期异常都包装成 500，且不泄漏内部栈信息。"""
    patch_run_chat_turn["error"] = RuntimeError("boom internal")

    resp = client.post(
        "/chat",
        json={"session_id": "u1", "message": "hi"},
    )
    assert resp.status_code == 500
    # detail 是预设的通用文案，不回显原始异常
    assert resp.json()["detail"] == "agent execution failed"


def test_chat_missing_required_field_returns_422(client: TestClient) -> None:
    """Pydantic 校验层：缺字段直接 422，根本走不到 chat_runner。"""
    resp = client.post("/chat", json={"session_id": "u1"})  # 缺 message
    assert resp.status_code == 422


def test_import_knowledge_endpoint_returns_index_summary(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes as routes_mod

    def fake_import(payload):
        assert payload.content == "WAI-ARIA 是无障碍技术规范。"
        return KnowledgeImportResult(
            doc_id="doc-api",
            title="API 导入文档",
            source="api.md",
            source_type="md",
            content_hash="hash",
            chunk_count=1,
            indexed_to_sqlite=True,
            indexed_to_chroma=True,
        )

    monkeypatch.setattr(routes_mod, "import_knowledge_document", fake_import)

    resp = client.post(
        "/knowledge/import",
        json={
            "title": "API 导入文档",
            "source": "api.md",
            "source_type": "md",
            "content": "WAI-ARIA 是无障碍技术规范。",
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["doc_id"] == "doc-api"
    assert body["chunk_count"] == 1
    assert body["indexed_to_sqlite"] is True
    assert body["indexed_to_chroma"] is True


def test_import_knowledge_file_endpoint_returns_index_summary(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes as routes_mod

    def fake_import(payload):
        assert payload.content == "# WAI-ARIA\n\nWAI-ARIA 是无障碍技术规范。"
        assert payload.title == "上传文档"
        assert payload.source == "upload.md"
        assert payload.source_type == "md"
        assert payload.metadata == {"topic": "a11y"}
        return KnowledgeImportResult(
            doc_id="doc-upload",
            title="上传文档",
            source="upload.md",
            source_type="md",
            content_hash="hash",
            chunk_count=1,
            indexed_to_sqlite=True,
            indexed_to_chroma=True,
        )

    monkeypatch.setattr(routes_mod, "import_knowledge_document", fake_import)

    resp = client.post(
        "/knowledge/import/file",
        data={
            "title": "上传文档",
            "metadata_json": '{"topic":"a11y"}',
        },
        files={
            "file": (
                "upload.md",
                b"# WAI-ARIA\n\nWAI-ARIA \xe6\x98\xaf\xe6\x97\xa0\xe9\x9a\x9c\xe7\xa2\x8d\xe6\x8a\x80\xe6\x9c\xaf\xe8\xa7\x84\xe8\x8c\x83\xe3\x80\x82",
                "text/markdown",
            )
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["doc_id"] == "doc-upload"
    assert body["source_type"] == "md"
    assert body["indexed_to_chroma"] is True


def test_delete_knowledge_doc_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes as routes_mod

    monkeypatch.setattr(
        routes_mod,
        "delete_knowledge_document",
        lambda doc_id: KnowledgeDeleteResult(
            doc_id=doc_id,
            deleted=True,
            chunk_count=2,
            deleted_from_sqlite=True,
            deleted_from_chroma=True,
        ),
    )

    resp = client.delete("/knowledge/docs/doc-api")

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "doc-api"
    assert body["deleted"] is True
    assert body["deleted_from_chroma"] is True


def test_delete_knowledge_doc_endpoint_returns_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes as routes_mod

    monkeypatch.setattr(
        routes_mod,
        "delete_knowledge_document",
        lambda doc_id: KnowledgeDeleteResult(
            doc_id=doc_id,
            deleted=False,
            chunk_count=0,
            deleted_from_sqlite=False,
            deleted_from_chroma=False,
        ),
    )

    resp = client.delete("/knowledge/docs/missing")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "document not found"


def test_reindex_knowledge_doc_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes as routes_mod

    monkeypatch.setattr(
        routes_mod,
        "reindex_knowledge_document",
        lambda doc_id: KnowledgeReindexResult(
            doc_id=doc_id,
            doc_count=1,
            chunk_count=3,
            reindexed_to_chroma=True,
        ),
    )

    resp = client.post("/knowledge/docs/doc-api/reindex")

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "doc-api"
    assert body["chunk_count"] == 3
    assert body["reindexed_to_chroma"] is True


def test_reindex_all_knowledge_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes as routes_mod

    monkeypatch.setattr(
        routes_mod,
        "reindex_all_knowledge_documents",
        lambda: KnowledgeReindexResult(
            doc_id="*",
            doc_count=4,
            chunk_count=10,
            reindexed_to_chroma=True,
        ),
    )

    resp = client.post("/knowledge/reindex")

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "*"
    assert body["doc_count"] == 4
    assert body["chunk_count"] == 10


def test_inspect_knowledge_doc_chunks_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes as routes_mod

    class FakeCatalog:
        def get_document(self, doc_id: str) -> dict:
            assert doc_id == "doc-api"
            return {"doc_id": doc_id}

    monkeypatch.setattr(routes_mod, "KnowledgeCatalog", lambda: FakeCatalog())
    monkeypatch.setattr(
        routes_mod,
        "inspect_document_chunks",
        lambda doc_id, **kwargs: ChunkQualityReport(
            doc_id=doc_id,
            chunk_count=2,
            total_chars=180,
            min_chars=80,
            max_chars=100,
            avg_chars=90.0,
            median_chars=90.0,
            short_chunk_count=0,
            long_chunk_count=0,
            section_count=1,
            top_sections=[{"section_title": "Intro", "chunk_count": 2}],
            samples=[{"chunk_id": "doc-api::chunk::0", "preview": "hello"}],
            warnings=["low_section_diversity"],
        ),
    )

    resp = client.get("/knowledge/docs/doc-api/chunks/inspect?sample_limit=1")

    assert resp.status_code == 200, resp.text
    report = resp.json()["report"]
    assert report["doc_id"] == "doc-api"
    assert report["chunk_count"] == 2
    assert report["top_sections"][0]["section_title"] == "Intro"
    assert report["warnings"] == ["low_section_diversity"]


def test_inspect_knowledge_doc_chunks_endpoint_returns_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes as routes_mod

    class FakeCatalog:
        def get_document(self, doc_id: str) -> None:
            return None

    monkeypatch.setattr(routes_mod, "KnowledgeCatalog", lambda: FakeCatalog())

    resp = client.get("/knowledge/docs/missing/chunks/inspect")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "document not found"


def _search_report(query: str) -> SearchInspectReport:
    return SearchInspectReport(
        query=query,
        query_type="definition",
        query_classification={"type": "definition", "confidence": 0.9},
        pipeline_config={"doc_top_k": 6},
        retrieval_debug={"dense_count": 1, "lexical_count": 1},
        stage_metrics={"counts": {"dense": 1, "lexical": 1}},
        timings_ms={"docSearch": 1.0},
        dense_hits=[{"id": "doc1::chunk::0"}],
        lexical_hits=[{"id": "doc1::chunk::0"}],
        hybrid_hits=[{"id": "doc1::chunk::0"}],
        returned_hits=[{"id": "doc1::chunk::0"}],
        filtered_hits=[{"id": "doc1::chunk::0"}],
        reranked_hits=[{"id": "doc1::chunk::0"}],
        merged_hits=[{"id": "doc1::chunk::0"}],
        citations=[{"ref": "[1]", "doc_id": "doc1"}],
        context_preview="WAI-ARIA 是无障碍技术规范。",
        context_chars=20,
        context_compression={"enabled": True},
        errors=[],
    )


def test_inspect_knowledge_search_post_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes as routes_mod

    monkeypatch.setattr(
        routes_mod,
        "inspect_retrieval",
        lambda query, **kwargs: _search_report(query),
    )

    resp = client.post(
        "/knowledge/search/inspect",
        json={"query": "WAI-ARIA 是什么", "top_k": 3},
    )

    assert resp.status_code == 200, resp.text
    report = resp.json()["report"]
    assert report["query"] == "WAI-ARIA 是什么"
    assert report["query_type"] == "definition"
    assert report["retrieval_debug"]["dense_count"] == 1
    assert report["stage_metrics"]["counts"]["dense"] == 1


def test_inspect_knowledge_search_get_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes as routes_mod

    monkeypatch.setattr(
        routes_mod,
        "inspect_retrieval",
        lambda query, **kwargs: _search_report(query),
    )

    resp = client.get("/knowledge/search/inspect?query=Skills%20是什么&top_k=3")

    assert resp.status_code == 200, resp.text
    assert resp.json()["report"]["query"] == "Skills 是什么"


def test_inspect_knowledge_search_post_endpoint_returns_400(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes as routes_mod

    def raise_value_error(query: str, **kwargs):
        raise ValueError("query must not be empty")

    monkeypatch.setattr(routes_mod, "inspect_retrieval", raise_value_error)

    resp = client.post("/knowledge/search/inspect", json={"query": "   "})

    assert resp.status_code == 400
    assert resp.json()["detail"] == "query must not be empty"


def test_preview_knowledge_doc_rechunk_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes as routes_mod

    current = ChunkQualityReport(
        doc_id="doc-api",
        chunk_count=2,
        total_chars=180,
        min_chars=80,
        max_chars=100,
        avg_chars=90.0,
        median_chars=90.0,
        short_chunk_count=0,
        long_chunk_count=0,
        section_count=1,
        top_sections=[],
        samples=[],
        warnings=[],
    )
    preview = ChunkQualityReport(
        doc_id="doc-api",
        chunk_count=3,
        total_chars=180,
        min_chars=50,
        max_chars=70,
        avg_chars=60.0,
        median_chars=60.0,
        short_chunk_count=0,
        long_chunk_count=0,
        section_count=1,
        top_sections=[],
        samples=[],
        warnings=[],
    )

    def fake_preview(doc_id: str, **kwargs):
        assert doc_id == "doc-api"
        return RechunkPreviewReport(
            doc_id=doc_id,
            title="API 文档",
            source="api.md",
            source_type="md",
            applied=False,
            source_mode="reconstructed_from_chunks",
            params={"chunk_size_chars": 120},
            current=current,
            preview=preview,
            delta={"chunk_count": 1},
            warnings=[],
        )

    monkeypatch.setattr(routes_mod, "preview_rechunk_document", fake_preview)

    resp = client.post(
        "/knowledge/docs/doc-api/rechunk/preview",
        json={
            "chunk_size_chars": 120,
            "chunk_overlap_chars": 20,
            "min_chunk_chars": 20,
        },
    )

    assert resp.status_code == 200, resp.text
    report = resp.json()["report"]
    assert report["doc_id"] == "doc-api"
    assert report["applied"] is False
    assert report["preview"]["chunk_count"] == 3
    assert report["delta"]["chunk_count"] == 1


def test_preview_knowledge_doc_rechunk_endpoint_returns_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes as routes_mod

    def fake_preview(doc_id: str, **kwargs):
        raise ValueError("document not found")

    monkeypatch.setattr(routes_mod, "preview_rechunk_document", fake_preview)

    resp = client.post("/knowledge/docs/missing/rechunk/preview", json={})

    assert resp.status_code == 404
    assert resp.json()["detail"] == "document not found"


def test_chat_persists_session_state_across_turns(
    client: TestClient, patch_run_chat_turn: dict
) -> None:
    """两轮请求：第二轮应能从 session_store 拿到上一轮的 messages。"""
    # 第 1 轮
    client.post("/chat", json={"session_id": "u1", "message": "first"})
    # 调整 stub 返回，模拟新一轮的 answer
    patch_run_chat_turn["result"] = {
        **patch_run_chat_turn["result"],
        "answer": "second answer",
        "summary": "summary-v2",
    }
    # 第 2 轮
    client.post("/chat", json={"session_id": "u1", "message": "second"})

    # 第 2 次调用时 state 里应带着第 1 次 commit 的 messages / summary
    second_state = patch_run_chat_turn["calls"][1]["state"]
    assert len(second_state["messages"]) == 2  # 上一轮的 user + assistant
    assert second_state["messages"][0]["content"] == "first"
    assert second_state["messages"][1]["content"] == "stub answer"

    # session_store 里最终保存的是第 2 轮结果
    final = session_store["u1"]
    assert final["summary"] == "summary-v2"
    assert final["messages"][-1]["content"] == "second answer"


def test_chat_different_sessions_are_isolated(
    client: TestClient, patch_run_chat_turn: dict
) -> None:
    client.post("/chat", json={"session_id": "alice", "message": "hi"})
    client.post("/chat", json={"session_id": "bob", "message": "hey"})

    assert "alice" in session_store
    assert "bob" in session_store
    # 两个 session 的 messages 互不干扰
    assert session_store["alice"]["messages"][0]["content"] == "hi"
    assert session_store["bob"]["messages"][0]["content"] == "hey"


# ---------------------------------------------------------------------------
# POST /chat/stream
# ---------------------------------------------------------------------------


def _parse_sse(raw: str) -> list[dict[str, Any]]:
    """把一段 SSE 原始文本切成事件列表。

    - 忽略 `: ping` 心跳注释行
    - 每个 event 至少含 `event` / `data`，data 走 json.loads
    """
    events: list[dict[str, Any]] = []
    for frame in raw.split("\n\n"):
        frame = frame.strip()
        if not frame or frame.startswith(":"):
            continue
        event: dict[str, Any] = {}
        for line in frame.splitlines():
            if line.startswith("event:"):
                event["event"] = line[len("event:") :].strip()
            elif line.startswith("data:"):
                event["data"] = json.loads(line[len("data:") :].strip())
        if event:
            events.append(event)
    return events


def test_chat_stream_emits_start_chunk_done(
    client: TestClient, patch_run_chat_turn: dict
) -> None:
    """底层未流式返回时，streaming.py 应把答案切成 chunk 兜底推给前端。"""
    # 故意让 answer 长于 chunk_size=24，确保切出至少 2 片
    long_answer = "A" * 30 + "B" * 30
    patch_run_chat_turn["result"] = {
        **patch_run_chat_turn["result"],
        "answer": long_answer,
        "streamed_answer": False,
    }

    with client.stream(
        "POST",
        "/chat/stream",
        json={"session_id": "s1", "message": "hi"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = "".join(resp.iter_text())

    events = _parse_sse(raw)
    event_names = [e["event"] for e in events]

    # 必须有 start / 多个 chunk / done，且顺序正确
    assert event_names[0] == "start"
    assert "done" in event_names
    assert event_names.index("start") < event_names.index("done")

    chunks = [e for e in events if e["event"] == "chunk"]
    assert len(chunks) >= 2
    # chunk 拼起来等于原答案
    assembled = "".join(c["data"]["delta"] for c in chunks)
    assert assembled == long_answer

    # done 事件里应带完整 payload（含 answer / routes）
    done_event = next(e for e in events if e["event"] == "done")
    assert done_event["data"]["answer"] == long_answer
    assert done_event["data"]["routes"] == ["chat_agent"]


def test_chat_stream_skips_chunk_fallback_when_already_streamed(
    client: TestClient, patch_run_chat_turn: dict
) -> None:
    """底层声称已流式发送（streamed_answer=True）时，API 层不应再 chunk 兜底。"""
    patch_run_chat_turn["result"] = {
        **patch_run_chat_turn["result"],
        "answer": "A" * 100,
        "streamed_answer": True,
    }

    with client.stream(
        "POST",
        "/chat/stream",
        json={"session_id": "s1", "message": "hi"},
    ) as resp:
        raw = "".join(resp.iter_text())

    events = _parse_sse(raw)
    # 不应该有任何 chunk 事件
    assert not any(e["event"] == "chunk" for e in events)
    # 但 start / done 仍要完整
    assert any(e["event"] == "start" for e in events)
    assert any(e["event"] == "done" for e in events)


def test_chat_stream_error_becomes_error_event_not_http_500(
    client: TestClient, patch_run_chat_turn: dict
) -> None:
    """worker 线程里抛异常时，HTTP 仍然是 200，错误通过 SSE error 事件发给前端。"""
    patch_run_chat_turn["error"] = RuntimeError("boom")

    with client.stream(
        "POST",
        "/chat/stream",
        json={"session_id": "s1", "message": "hi"},
    ) as resp:
        # SSE 流本身返回 200，错误在事件流里
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())

    events = _parse_sse(raw)
    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) == 1
    # chat_runner 把未知异常包成 HTTPException(500, "agent execution failed")
    assert error_events[0]["data"]["status_code"] == 500
    assert error_events[0]["data"]["detail"] == "agent execution failed"

    # 不应发 done
    assert not any(e["event"] == "done" for e in events)


def test_chat_stream_empty_session_id_sends_error_event(
    client: TestClient, patch_run_chat_turn: dict
) -> None:
    """空 session_id 会在 chat_runner 里抛 HTTPException(400)，worker 翻成 error 事件。"""
    with client.stream(
        "POST",
        "/chat/stream",
        json={"session_id": "  ", "message": "hi"},
    ) as resp:
        raw = "".join(resp.iter_text())

    events = _parse_sse(raw)
    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["data"]["status_code"] == 400
    assert "session_id" in error_events[0]["data"]["detail"]

"""`rag_agent_node` 测试。

分两层：
1. **pure/LLM-only 层**：`generate_answer_for_context` 的三个分支（doc / memory /
   资料不足）和异常降级。直接用 `llm_stub` 接管 LLM，无需任何 Chroma / embedding。
2. **node 编排层**：把 `rewrite_rag_query` / `retrieve_docs_for_rag` /
   `retrieve_memory_for_rag` 三个子阶段直接 monkeypatch 成返回固定 dataclass，
   这样绕开 embedding + Chroma + LLM rerank 的真实 I/O，只测 `rag_agent_node` 的
   编排逻辑（什么时候走 memory 兜底、什么时候返回兜底文案、errors 怎么传递）。

不 mock Chroma / embedding 的理由：
- 它们上面还有 doc_pipeline / memory_pipeline 两个"结果对象"粒度的边界，
  在那里打桩更稳定；一旦 Chroma API 变化不会把测试一次性打爆
- pure-compute 的 `merge_doc_hits` / `rank_hybrid` 已经在 test_retrieval_doc.py 覆盖
"""

from __future__ import annotations

from app.agents.rag.answer import generate_answer_for_context
from app.agents.rag.types import (
    DocRetrievalResult,
    MemoryRetrievalResult,
    RagContext,
    RewriteResult,
)
from app.constants.routes import ROUTE_RAG_AGENT

# ------------------------ generate_answer_for_context ------------------------


def _strategy() -> dict:
    return {
        "name": "default_short",
        "answer_style": "简短回答",
        "context_chars": 2000,
        "max_tokens": 500,
    }


def _ctx(doc: str = "", mem: str = "") -> RagContext:
    return RagContext(
        context=(doc + "\n" + mem).strip(),
        doc_context=doc,
        memory_context=mem,
    )


def test_answer_for_context_uses_doc_branch_when_strong_knowledge(
    llm_stub,
) -> None:
    llm_stub.set_response("doc-answer")

    result = generate_answer_for_context(
        question="WAI-ARIA 是什么",
        rag_context=_ctx(doc="doc-chunk-1"),
        doc_answer_strategy=_strategy(),
        has_strong_knowledge=True,
        has_memory=False,
    )

    assert result.answer == "doc-answer"
    assert result.errors == []
    # doc 分支应带 max_completion_tokens（来自 strategy）
    assert len(llm_stub.calls) == 1
    call = llm_stub.calls[0]
    assert call["max_completion_tokens"] == 500
    # user prompt 里带上 question + doc context
    user_content = call["messages"][1]["content"]
    assert "WAI-ARIA" in user_content
    assert "doc-chunk-1" in user_content


def test_answer_for_context_uses_memory_branch_when_no_strong_knowledge(
    llm_stub,
) -> None:
    llm_stub.set_response("memory-answer")

    result = generate_answer_for_context(
        question="我们之前聊过什么",
        rag_context=_ctx(mem="memo-chunk"),
        doc_answer_strategy=_strategy(),
        has_strong_knowledge=False,
        has_memory=True,
    )

    assert result.answer == "memory-answer"
    # memory 分支 max_tokens 来自 RAG_CONFIG.max_memory_answer_tokens（不是 strategy 的 500）
    assert llm_stub.calls[0]["max_completion_tokens"] != 500
    user_content = llm_stub.calls[0]["messages"][1]["content"]
    assert "memo-chunk" in user_content


def test_answer_for_context_returns_insufficient_when_nothing(llm_stub) -> None:
    # 既无 doc 也无 memory → 直接兜底文案，不该调 LLM
    result = generate_answer_for_context(
        question="q",
        rag_context=_ctx(),
        doc_answer_strategy=_strategy(),
        has_strong_knowledge=False,
        has_memory=False,
    )
    assert result.answer == "资料不足"
    assert result.errors == []
    assert llm_stub.calls == []


def test_answer_for_context_llm_exception_degrades_gracefully(
    llm_stub,
) -> None:
    def explode(**_):
        raise RuntimeError("LLM down")

    llm_stub.set_response_fn(explode)

    result = generate_answer_for_context(
        question="q",
        rag_context=_ctx(doc="d"),
        doc_answer_strategy=_strategy(),
        has_strong_knowledge=True,
        has_memory=False,
    )
    assert result.answer == "知识检索暂时失败，请稍后再试。"
    assert result.errors  # 记录了错误信息，上游拿到 debug_info 能看到


# ----------------------------- rag_agent_node 编排 -----------------------------


def _empty_doc_result() -> DocRetrievalResult:
    return DocRetrievalResult(
        docs=[],
        filtered_docs=[],
        doc_hits=[],
        merged_doc_hits=[],
        retrieval_debug={},
        errors=[],
        timings_ms={"dense": 0.0, "keyword": 0.0},
    )


def _doc_result_with_hit() -> DocRetrievalResult:
    hit = {"id": "doc-1", "content": "WAI-ARIA 是一套无障碍规范", "score": 0.9}
    return DocRetrievalResult(
        docs=[hit],
        filtered_docs=[hit],
        doc_hits=[hit],
        merged_doc_hits=[hit],
        retrieval_debug={},
        errors=[],
        timings_ms={"dense": 1.0, "keyword": 0.5},
    )


def _empty_memory_result() -> MemoryRetrievalResult:
    return MemoryRetrievalResult(
        memory_hits=[],
        memory_before_rerank=[],
        retrieval_debug={},
        errors=[],
        timings_ms={},
    )


def _memory_result_with_hit() -> MemoryRetrievalResult:
    hit = {"id": "mem-1", "content": "用户之前住在北京", "score": 0.8}
    return MemoryRetrievalResult(
        memory_hits=[hit],
        memory_before_rerank=[hit],
        retrieval_debug={},
        errors=[],
        timings_ms={"memory": 1.0},
    )


def _patch_pipelines(
    monkeypatch,
    *,
    rewritten: str = "rewritten-q",
    rewrite_errors: list[str] | None = None,
    doc_result: DocRetrievalResult | None = None,
    memory_result: MemoryRetrievalResult | None = None,
) -> dict:
    """把 rewrite / doc / memory 三个子阶段打桩到 rag_agent_node 所在模块。

    返回一个 dict 用来断言：memory_enabled 记录 retrieve_memory_for_rag 的 enabled 入参。
    """
    import app.agents.rag.node as node_mod

    captured = {"memory_enabled": None}

    def fake_rewrite(message, messages, summary):
        return RewriteResult(
            query=rewritten,
            errors=list(rewrite_errors or []),
            timing_ms=1.0,
        )

    def fake_docs(_q):
        return doc_result or _empty_doc_result()

    def fake_memory(_q, *, session_id, enabled):
        captured["memory_enabled"] = enabled
        return memory_result or _empty_memory_result()

    monkeypatch.setattr(node_mod, "rewrite_rag_query", fake_rewrite)
    monkeypatch.setattr(node_mod, "retrieve_docs_for_rag", fake_docs)
    monkeypatch.setattr(node_mod, "retrieve_memory_for_rag", fake_memory)

    return captured


def _rag_state(message: str = "什么是 WAI-ARIA") -> dict:
    return {
        "messages": [{"role": "user", "content": message}],
        "summary": "",
        "session_id": "test",
    }


def test_rag_node_happy_path_uses_doc_answer(monkeypatch, llm_stub) -> None:
    captured = _patch_pipelines(
        monkeypatch,
        doc_result=_doc_result_with_hit(),
    )
    llm_stub.set_response("doc-based-answer")

    from app.agents.rag.node import rag_agent_node

    result = rag_agent_node(_rag_state())

    assert result["answer"] == "doc-based-answer"
    assert result["agent_outputs"][ROUTE_RAG_AGENT] == "doc-based-answer"
    assert result["rewritten_query"] == "rewritten-q"
    # 文档命中充足时，memory 检索必须被禁用（省成本）
    assert captured["memory_enabled"] is False

    debug = result["debug_info"][ROUTE_RAG_AGENT]
    assert debug["doc_used"] is True
    assert debug["memory_used"] is False


def test_rag_node_falls_back_to_memory_when_no_docs(monkeypatch, llm_stub) -> None:
    captured = _patch_pipelines(
        monkeypatch,
        doc_result=_empty_doc_result(),
        memory_result=_memory_result_with_hit(),
    )
    llm_stub.set_response("memory-based-answer")

    from app.agents.rag.node import rag_agent_node

    result = rag_agent_node(_rag_state("我们之前聊过北京"))

    assert result["answer"] == "memory-based-answer"
    # 无文档时 memory 检索必须启用
    assert captured["memory_enabled"] is True
    debug = result["debug_info"][ROUTE_RAG_AGENT]
    assert debug["doc_used"] is False
    assert debug["memory_used"] is True


def test_rag_node_no_knowledge_returns_insufficient(monkeypatch, llm_stub) -> None:
    _patch_pipelines(
        monkeypatch,
        doc_result=_empty_doc_result(),
        memory_result=_empty_memory_result(),
    )

    from app.agents.rag.node import rag_agent_node

    result = rag_agent_node(_rag_state())

    assert result["answer"] == "资料不足"
    # 兜底路径不应调 LLM
    assert llm_stub.calls == []


def test_rag_node_propagates_subphase_errors_to_debug(monkeypatch, llm_stub) -> None:
    # rewrite 阶段报错 → errors 应出现在 debug_info 里，主链路不中断
    _patch_pipelines(
        monkeypatch,
        rewrite_errors=["rewrite_failed: timeout"],
        doc_result=_empty_doc_result(),
        memory_result=_empty_memory_result(),
    )

    from app.agents.rag.node import rag_agent_node

    result = rag_agent_node(_rag_state())

    debug = result["debug_info"][ROUTE_RAG_AGENT]
    assert any("rewrite_failed" in err for err in debug["errors"])
    # 回答仍应正常兜底
    assert result["answer"] == "资料不足"

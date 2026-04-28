import os
from contextvars import ContextVar
from typing import Any

from langsmith import get_current_run_tree

from app.utils.logger import logger

LANGSMITH_TRACING_ENV = "LANGSMITH_TRACING"
LANGSMITH_API_KEY_ENV = "LANGSMITH_API_KEY"
LANGSMITH_PROJECT_ENV = "LANGSMITH_PROJECT"
LANGSMITH_ENDPOINT_ENV = "LANGSMITH_ENDPOINT"

DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
DEFAULT_LANGSMITH_PROJECT = "langgraph-agent-dev"
TRACE_MESSAGE_PREVIEW_CHARS = 120
TRACE_DOC_PREVIEW_CHARS = 80
TRACE_MAX_TOP_DOCS = 3
_MODEL_CALL_INDEX: ContextVar[int] = ContextVar("model_call_index", default=0)


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _preview(text: str, limit: int = TRACE_MESSAGE_PREVIEW_CHARS) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def is_langsmith_tracing_enabled() -> bool:
    """判断当前进程是否开启 LangSmith tracing。

    LangSmith 的 Python SDK 会读取 LANGSMITH_* 环境变量完成真正上报。
    这里仅用于 debug 展示和本地配置自检，不主动打印 API Key。
    """

    return _env_enabled(LANGSMITH_TRACING_ENV, False)


def get_langsmith_runtime_info() -> dict:
    """返回可安全暴露到 debug 的 LangSmith 运行信息。"""

    return {
        "enabled": is_langsmith_tracing_enabled(),
        "project": os.getenv(LANGSMITH_PROJECT_ENV, DEFAULT_LANGSMITH_PROJECT),
        "endpoint": os.getenv(LANGSMITH_ENDPOINT_ENV, DEFAULT_LANGSMITH_ENDPOINT),
        "api_key_configured": bool(os.getenv(LANGSMITH_API_KEY_ENV, "").strip()),
    }


def build_graph_trace_config(state: dict, message: str) -> dict:
    """构造 LangGraph invoke 的 tracing config。

    tags/metadata 会出现在 LangSmith trace 里，方便按请求、session、入口类型筛选。
    不放完整用户输入，避免 trace 里出现过长或敏感内容。
    """

    request_id = state.get("request_id", "")
    session_id = state.get("session_id", "default")
    return {
        "run_name": "chat_turn",
        "tags": ["api_chat", f"session:{session_id}"],
        "metadata": {
            "request_id": request_id,
            "session_id": session_id,
            "debug": bool(state.get("debug", False)),
            "message_preview": _preview(message),
            "conversation_history_path_set": bool(
                state.get("conversation_history_path", "")
            ),
        },
        # thread_id 让 LangGraph/LangSmith 更容易把同一 session 的运行串起来。
        "configurable": {
            "thread_id": session_id,
        },
    }


def reset_model_call_index() -> None:
    """重置当前请求内的模型调用序号。"""

    _MODEL_CALL_INDEX.set(0)


def next_model_call_index() -> int:
    """递增并返回当前请求内的模型调用序号。"""

    current = _MODEL_CALL_INDEX.get()
    next_index = current + 1
    _MODEL_CALL_INDEX.set(next_index)
    return next_index


def _compact_doc_hit(hit: dict) -> dict:
    """压缩文档命中信息，避免把大段原文写进 trace metadata。"""

    return {
        "id": hit.get("id", ""),
        "doc_id": hit.get("doc_id", ""),
        "chunk_index": hit.get("chunk_index"),
        "score": hit.get("score"),
        "semantic_score": hit.get("semantic_score"),
        "keyword_score_norm": hit.get("keyword_score_norm"),
        "retrieval_sources": hit.get("retrieval_sources", []),
        "preview": _preview(hit.get("preview", ""), TRACE_DOC_PREVIEW_CHARS),
    }


def build_rag_trace_metadata(rag_debug: dict[str, Any]) -> dict:
    """从 rag_agent debug 中抽取适合 LangSmith 检索/筛选的指标。"""

    top_docs = rag_debug.get("top_docs", [])
    filtered_docs = rag_debug.get("filtered_docs", [])
    merged_docs = rag_debug.get("merged_docs", [])
    citations = rag_debug.get("citations", [])
    retrieval_debug = rag_debug.get("retrieval_debug", {})
    doc_retrieval = retrieval_debug.get("doc", {})
    memory_retrieval = retrieval_debug.get("memory", {})
    sub_timings_ms = rag_debug.get("sub_timings_ms", {})

    return {
        "rag.doc_used": bool(rag_debug.get("doc_used", False)),
        "rag.memory_used": bool(rag_debug.get("memory_used", False)),
        "rag.query_type": rag_debug.get("query_type", ""),
        "rag.answer_strategy": rag_debug.get("answer_strategy", ""),
        "rag.threshold": rag_debug.get("threshold"),
        "rag.doc_context_chars": rag_debug.get("doc_context_chars", 0),
        "rag.answer_context_chars": rag_debug.get("answer_context_chars", 0),
        "rag.answer_max_tokens": rag_debug.get("answer_max_tokens", 0),
        "rag.top_doc_score": top_docs[0].get("score") if top_docs else None,
        "rag.filtered_count": len(filtered_docs),
        "rag.merged_count": len(merged_docs),
        "rag.citation_count": len(citations),
        "rag.citation_doc_ids": [
            citation.get("doc_id", "")
            for citation in citations
            if citation.get("doc_id")
        ],
        "rag.doc_returned_count": doc_retrieval.get("returned_count", 0),
        "rag.doc_consumed_count": doc_retrieval.get("consumed_count", 0),
        "rag.rerank_skipped": bool(doc_retrieval.get("rerank_skipped", False)),
        "rag.rerank_skip_reason": doc_retrieval.get("rerank_skip_reason", ""),
        "rag.memory_search_enabled": bool(memory_retrieval.get("enabled", False)),
        "rag.memory_skip_reason": memory_retrieval.get("skip_reason", ""),
        "rag.timing.doc_search_ms": sub_timings_ms.get("docSearch", 0),
        "rag.timing.doc_rerank_ms": sub_timings_ms.get("docRerank", 0),
        "rag.timing.memory_search_ms": sub_timings_ms.get("memorySearch", 0),
        "rag.timing.memory_rerank_ms": sub_timings_ms.get("memoryRerank", 0),
        "rag.timing.answer_generation_ms": sub_timings_ms.get("answerGeneration", 0),
        "rag.top_docs": [
            _compact_doc_hit(hit) for hit in top_docs[:TRACE_MAX_TOP_DOCS]
        ],
    }


def build_model_trace_metadata(
    *,
    kind: str,
    profile: str,
    provider: str,
    model: str,
    stage: str = "",
    call_index: int = 0,
    streaming: bool = False,
    input_count: int = 0,
    timeout_seconds: float = 0.0,
    max_retries: int = 0,
) -> dict:
    """构造模型调用 trace metadata。

    只记录 profile/provider/model 等运行时信息，不记录 prompt 正文。
    prompt 本身 LangSmith 已能在 LLM run input 里查看，metadata 用来做筛选聚合。
    """

    return {
        "model.kind": kind,
        "model.profile": profile,
        "model.provider": provider,
        "model.name": model,
        "model.stage": stage,
        "model.call_index": call_index,
        "model.streaming": streaming,
        "model.input_count": input_count,
        "model.timeout_seconds": timeout_seconds,
        "model.max_retries": max_retries,
    }


def add_current_run_metadata(
    metadata: dict[str, Any],
    tags: list[str] | None = None,
    event_name: str = "",
) -> None:
    """给当前 LangSmith run 追加 metadata/tags/event。

    tracing 未开启或当前没有 run context 时直接 no-op。
    这里不能影响主链路，所以所有异常都吞掉。
    """

    if not is_langsmith_tracing_enabled():
        logger.debug(
            "tracing.skip",
            extra={"reason": "disabled", "event": event_name},
        )
        return

    try:
        run_tree = get_current_run_tree()
        if run_tree is None:
            logger.debug(
                "tracing.skip",
                extra={"reason": "no_run_context", "event": event_name},
            )
            return
        if metadata:
            run_tree.add_metadata(metadata)
        if tags:
            run_tree.add_tags(tags)
        if event_name:
            run_tree.add_event({"name": event_name, "metadata": metadata})
        logger.debug(
            "tracing.applied",
            extra={
                "event": event_name,
                "metadata_keys": list(metadata.keys()) if metadata else [],
                "tag_count": len(tags) if tags else 0,
            },
        )
    except Exception as exc:
        logger.debug(
            "tracing.failed",
            extra={"event": event_name, "error": repr(exc)},
        )
        return

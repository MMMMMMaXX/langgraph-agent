"""聊天执行核心：把 HTTP 请求转成一次 graph.invoke 并组织响应。

职责单一化：这里 **不** 管 FastAPI 路由、**不** 管 SSE 格式，
只做 “验证 → 加锁 → 快照 → 执行 → 提交 → 组装响应”。
路由层（routes.py）和流式层（streaming.py）复用这个函数。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

from app.chat_service import run_chat_turn
from app.tracing import get_langsmith_runtime_info
from app.utils.logger import log_request, now_ms

from .schemas import ChatRequest, DebugPayload
from .session_store import (
    commit_session_state,
    get_session_lock,
    snapshot_session_state,
)

StreamCallback = Callable[[str, dict[str, Any]], None]


def _log_failure(
    request: ChatRequest,
    request_id: str,
    started_at_ms: float,
    error: str,
) -> None:
    """统一的失败日志记录，避免 3 处 except 分支重复 log_request 调用。"""

    log_request(
        stage="failed",
        request_id=request_id,
        session_id=request.session_id.strip() or "<empty>",
        message=request.message,
        duration_ms=now_ms() - started_at_ms,
        error=error,
    )


def _build_response_payload(
    request: ChatRequest,
    request_id: str,
    session_id: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """把 run_chat_turn 结果组装成 API 响应字典（debug 信息可选）。"""

    answer = result.get("answer", "")
    payload: dict[str, Any] = {
        "request_id": request_id,
        "session_id": session_id,
        "answer": answer,
        "routes": result.get("routes", []),
        "summary": result.get("summary", ""),
    }
    if request.debug:
        payload["debug"] = DebugPayload(
            node_timings=result.get("node_timings", {}),
            nodes=result.get("debug_info", {}),
            tracing={"langsmith": get_langsmith_runtime_info()},
        ).model_dump()
    return payload


def _invoke_with_session_lock(
    session_id: str,
    request: ChatRequest,
    request_id: str,
    stream_callback: StreamCallback | None,
) -> dict[str, Any]:
    """在 session_lock 保护下完成一次 turn。

    锁顺序：guard(瞬时取 session_lock) → session_lock(长时间) →
    guard(瞬时快照) → graph.invoke(无锁) → guard(瞬时写回)。
    任何改动必须保持这个顺序，否则有死锁风险，详见 session_store.py 的
    模块级 docstring。
    """

    session_lock = get_session_lock(session_id)
    with session_lock:
        state_snapshot = snapshot_session_state(session_id)
        current_state = {
            **state_snapshot,
            "request_id": request_id,
            "session_id": session_id,
            "debug": request.debug,
            "conversation_history_path": request.conversation_history_path.strip(),
            "stream_callback": stream_callback,
            "streamed_answer": False,
        }
        # 长时间操作：跑完整 graph，不持有任何全局锁。
        # stream_callback 可能被回调，但不会再触发 guard，安全。
        result = run_chat_turn(current_state, request.message)
        commit_session_state(session_id, result)
    return result


def build_chat_result(
    request: ChatRequest,
    request_id: str,
    stream_callback: StreamCallback | None = None,
) -> tuple[dict[str, Any], str, bool]:
    """执行一次聊天 turn，返回 (响应 payload, answer, streamed_answer)。

    调用方：
    - routes.chat            非流式返回 payload
    - streaming.run_worker   流式时需要额外拿 answer + streamed_answer 做 chunk 回退
    """

    normalized_session_id = request.session_id.strip()
    started_at_ms = now_ms()

    log_request(
        stage="started",
        request_id=request_id,
        session_id=normalized_session_id or "<empty>",
        message=request.message,
    )

    if not normalized_session_id:
        # 提前报错，避免进入锁流程
        raise HTTPException(status_code=400, detail="session_id must not be empty")

    try:
        result = _invoke_with_session_lock(
            normalized_session_id, request, request_id, stream_callback
        )
    except ValueError as exc:
        _log_failure(request, request_id, started_at_ms, str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        # 上层自行处理状态码，不当作 500。
        raise
    except Exception as exc:
        _log_failure(request, request_id, started_at_ms, str(exc))
        raise HTTPException(status_code=500, detail="agent execution failed") from exc

    payload = _build_response_payload(
        request, request_id, normalized_session_id, result
    )
    log_request(
        stage="completed",
        request_id=request_id,
        session_id=normalized_session_id,
        message=request.message,
        routes=payload["routes"],
        node_timings=result.get("node_timings", {}),
        answer=payload["answer"],
        duration_ms=now_ms() - started_at_ms,
    )
    return payload, payload["answer"], bool(result.get("streamed_answer", False))

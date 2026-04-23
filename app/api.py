import uuid
import json
import queue
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.chat_service import create_initial_state, run_chat_turn
from app.state import AgentState
from app.tracing import get_langsmith_runtime_info
from app.utils.logger import log_request, now_ms

app = FastAPI(title="LangGraph Agent API")
DEBUG_UI_PATH = Path(__file__).resolve().parent / "debug_ui.html"

session_store: dict[str, AgentState] = {}
session_store_guard = threading.Lock()
session_locks: dict[str, threading.RLock] = {}


def get_session_lock(session_id: str) -> threading.RLock:
    """获取单个 session 的互斥锁。

    同一 session 的请求必须串行执行，否则 `get -> run -> set` 会丢消息。
    不同 session 使用不同锁，仍然可以并发处理。
    """

    with session_store_guard:
        lock = session_locks.get(session_id)
        if lock is None:
            lock = threading.RLock()
            session_locks[session_id] = lock
        return lock


def clear_session_store() -> None:
    """清空内存 session 状态，主要供 eval / 测试使用。"""

    with session_store_guard:
        session_store.clear()
        session_locks.clear()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    debug: bool = False
    conversation_history_path: str = ""


class DebugPayload(BaseModel):
    node_timings: dict[str, float] = Field(default_factory=dict)
    nodes: dict[str, Any] = Field(default_factory=dict)
    tracing: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    answer: str
    routes: list[str] = []
    summary: str = ""
    debug: DebugPayload | None = None


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug-ui")
def debug_ui() -> FileResponse:
    return FileResponse(DEBUG_UI_PATH)


def get_or_create_session_state(session_id: str) -> AgentState:
    normalized_session_id = session_id.strip()
    if not normalized_session_id:
        raise ValueError("session_id must not be empty")

    with session_store_guard:
        if normalized_session_id not in session_store:
            session_store[normalized_session_id] = create_initial_state(
                normalized_session_id
            )

        return session_store[normalized_session_id]


def build_chat_result(
    request: ChatRequest,
    request_id: str,
    stream_callback=None,
) -> tuple[dict[str, Any], str, bool]:
    normalized_session_id = request.session_id.strip()
    started_at_ms = now_ms()

    log_request(
        stage="started",
        request_id=request_id,
        session_id=normalized_session_id or "<empty>",
        message=request.message,
    )

    try:
        session_lock = get_session_lock(normalized_session_id)
        with session_lock:
            state = get_or_create_session_state(normalized_session_id)
            current_state = {
                **state,
                "request_id": request_id,
                "session_id": normalized_session_id,
                "debug": request.debug,
                "conversation_history_path": request.conversation_history_path.strip(),
                "stream_callback": stream_callback,
                "streamed_answer": False,
            }
            result = run_chat_turn(current_state, request.message)
            with session_store_guard:
                session_store[normalized_session_id] = {
                    "session_id": normalized_session_id,
                    "messages": result.get("messages", []),
                    "summary": result.get("summary", ""),
                }
    except ValueError as exc:
        log_request(
            stage="failed",
            request_id=request_id,
            session_id=normalized_session_id or "<empty>",
            message=request.message,
            duration_ms=now_ms() - started_at_ms,
            error=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log_request(
            stage="failed",
            request_id=request_id,
            session_id=normalized_session_id or "<empty>",
            message=request.message,
            duration_ms=now_ms() - started_at_ms,
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail="agent execution failed") from exc

    answer = result.get("answer", "")
    routes = result.get("routes", [])
    summary = result.get("summary", "")
    node_timings = result.get("node_timings", {})
    debug_info = result.get("debug_info", {})
    streamed_answer = bool(result.get("streamed_answer", False))

    log_request(
        stage="completed",
        request_id=request_id,
        session_id=normalized_session_id,
        message=request.message,
        routes=routes,
        node_timings=node_timings,
        answer=answer,
        duration_ms=now_ms() - started_at_ms,
    )

    response_payload = {
        "request_id": request_id,
        "session_id": normalized_session_id,
        "answer": answer,
        "routes": routes,
        "summary": summary,
    }
    if request.debug:
        response_payload["debug"] = DebugPayload(
            node_timings=node_timings,
            nodes=debug_info,
            tracing={"langsmith": get_langsmith_runtime_info()},
        ).model_dump()

    return response_payload, answer, streamed_answer


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def chunk_text(text: str, chunk_size: int = 24) -> list[str]:
    if not text:
        return []
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    request_id = uuid.uuid4().hex[:12]
    response_payload, _, _ = build_chat_result(request, request_id)
    return ChatResponse(**response_payload)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    request_id = uuid.uuid4().hex[:12]
    event_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def worker() -> None:
        try:
            def stream_callback(event: str, data: dict[str, Any]) -> None:
                event_queue.put({"event": event, "data": data})

            response_payload, answer, streamed_answer = build_chat_result(
                request,
                request_id,
                stream_callback=stream_callback,
            )

            # 当前实现先保证 SSE 协议和前后端流式交互体验，
            # 即使底层 graph 还是同步执行，也能先发 start/ping，再按 chunk 推答案。
            if not streamed_answer:
                for chunk in chunk_text(answer):
                    event_queue.put(
                        {
                            "event": "chunk",
                            "data": {
                                "request_id": request_id,
                                "session_id": response_payload["session_id"],
                                "delta": chunk,
                            },
                        }
                    )

            event_queue.put(
                {
                    "event": "done",
                    "data": response_payload,
                }
            )
        except HTTPException as exc:
            event_queue.put(
                {
                    "event": "error",
                    "data": {
                        "request_id": request_id,
                        "status_code": exc.status_code,
                        "detail": exc.detail,
                    },
                }
            )
        except Exception as exc:
            event_queue.put(
                {
                    "event": "error",
                    "data": {
                        "request_id": request_id,
                        "status_code": 500,
                        "detail": str(exc),
                    },
                }
            )
        finally:
            event_queue.put({"event": "__end__", "data": {}})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def event_generator():
        yield sse_event(
            "start",
            {
                "request_id": request_id,
                "session_id": request.session_id.strip(),
            },
        )

        while True:
            try:
                item = event_queue.get(timeout=0.5)
            except queue.Empty:
                yield ": ping\n\n"
                continue

            if item["event"] == "__end__":
                break

            yield sse_event(item["event"], item["data"])

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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

# ---------------------------------------------------------------------------
# Session 并发模型
# ---------------------------------------------------------------------------
# session_store       : 全局 session 状态字典（内存）
# session_store_guard : 保护 session_store / session_locks 两张字典本身的
#                       "目录锁"，只用于瞬时 get/put 操作，不包含业务逻辑
# session_locks       : 每个 session 一把 RLock，保证同一 session 的请求串行
#                       （graph.invoke 期间持有，跨 session 请求仍可并发）
#
# 锁顺序约定（严格遵守，任何新增调用必须遵循）：
#   1) 先取 session_store_guard（短暂）查/建 session_lock，立即释放
#   2) 再取 session_lock 执行 graph.invoke（长时间）
#   3) 需要读写 session_store 时，在 session_lock 内再短暂取 guard
# 反向嵌套（先 session_lock 再 guard 长期持有）会造成死锁，禁止。
# ---------------------------------------------------------------------------
session_store: dict[str, AgentState] = {}
session_store_guard = threading.Lock()
session_locks: dict[str, threading.RLock] = {}


def get_session_lock(session_id: str) -> threading.RLock:
    """获取单个 session 的互斥锁（瞬时操作，不会阻塞业务）。

    只在 guard 内做字典查/建，立刻释放 guard，不持有任何长时间锁。
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


def _snapshot_session_state(session_id: str) -> AgentState:
    """在 guard 内拿到 session 状态的浅拷贝快照后立即释放 guard。

    返回的 dict 与 session_store 里的对象是不同引用，后续读写不会相互影响。
    """

    with session_store_guard:
        state = session_store.get(session_id)
        if state is None:
            state = create_initial_state(session_id)
            session_store[session_id] = state
        # 浅拷贝足够：后续 run_chat_turn 会基于这个 dict 构造新对象，
        # 不会原地修改 messages / summary 等字段。
        return dict(state)


def _commit_session_state(session_id: str, result: dict[str, Any]) -> None:
    """把 run_chat_turn 的结果写回 session_store（瞬时 guard）。"""

    with session_store_guard:
        session_store[session_id] = {
            "session_id": session_id,
            "messages": result.get("messages", []),
            "summary": result.get("summary", ""),
        }


def get_or_create_session_state(session_id: str) -> AgentState:
    """保留旧接口，供外部直接读取 session 状态（测试/脚本）。

    注意：返回的是 session_store 内的引用，调用方不应修改。
    业务路径请用 `_snapshot_session_state` 拿副本。
    """

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

    if not normalized_session_id:
        # 提前报错，避免进入锁流程
        raise HTTPException(status_code=400, detail="session_id must not be empty")

    try:
        # 1) 瞬时 guard：建/取 session_lock
        session_lock = get_session_lock(normalized_session_id)

        # 2) 持有 session_lock 执行整个 turn，guard 不再持有
        with session_lock:
            # 2a) 瞬时 guard：快照 session 状态
            state_snapshot = _snapshot_session_state(normalized_session_id)

            current_state = {
                **state_snapshot,
                "request_id": request_id,
                "session_id": normalized_session_id,
                "debug": request.debug,
                "conversation_history_path": request.conversation_history_path.strip(),
                "stream_callback": stream_callback,
                "streamed_answer": False,
            }

            # 2b) 长时间操作：跑完整 graph，不持有任何全局锁
            #     stream_callback 可能被回调，但不会再触发 guard，安全
            result = run_chat_turn(current_state, request.message)

            # 2c) 瞬时 guard：写回 session 状态
            _commit_session_state(normalized_session_id, result)
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
    except HTTPException:
        raise
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
    return [
        text[index : index + chunk_size] for index in range(0, len(text), chunk_size)
    ]


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

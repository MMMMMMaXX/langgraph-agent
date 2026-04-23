"""SSE 流式输出相关工具。

把三件事放在一起：
1. SSE event name 常量（避免字面量散落）
2. 纯字符串工具：sse_event 格式化、chunk_text 兜底分片
3. 工作线程 + event_generator，最终由 build_chat_stream_response 拼成
   StreamingResponse，让 routes.py 只用一行就能返回流式结果。

原来 chat_stream 路由函数内嵌 ~90 行闭包，这里抽成模块级函数后：
- worker 可以独立 mock event_queue 做单元测试
- 路由函数变成 3 行
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .chat_runner import build_chat_result
from .schemas import ChatRequest

# ---- SSE 事件名常量 ----
SSE_EVENT_START = "start"
SSE_EVENT_CHUNK = "chunk"
SSE_EVENT_DONE = "done"
SSE_EVENT_ERROR = "error"
# 内部哨兵：worker 通知 event_generator 结束循环，不发给前端。
SSE_EVENT_END = "__end__"

# 分片大小（字符数）。答案很短时不一定触发分片。
DEFAULT_STREAM_CHUNK_SIZE = 24
# event_generator 轮询队列的超时，用于在没有事件时发送 SSE 心跳注释行，
# 避免中间代理断开空闲连接。
QUEUE_POLL_TIMEOUT_SECONDS = 0.5


def sse_event(event: str, data: dict[str, Any]) -> str:
    """按 SSE 协议组装一条 event 帧。"""

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def chunk_text(text: str, chunk_size: int = DEFAULT_STREAM_CHUNK_SIZE) -> list[str]:
    """当底层 graph 没流式输出答案时，API 层按固定大小切片做兜底流式。"""

    if not text:
        return []
    return [
        text[index : index + chunk_size]
        for index in range(0, len(text), chunk_size)
    ]


def run_chat_stream_worker(
    request: ChatRequest,
    request_id: str,
    event_queue: "queue.Queue[dict[str, Any]]",
) -> None:
    """后台线程：真正跑 chat，把事件推给 event_queue。

    任何异常都会被翻译成 error 事件发给前端，最后必发一个 __end__ 哨兵，
    保证 event_generator 循环可以干净退出。
    """

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
                        "event": SSE_EVENT_CHUNK,
                        "data": {
                            "request_id": request_id,
                            "session_id": response_payload["session_id"],
                            "delta": chunk,
                        },
                    }
                )

        event_queue.put({"event": SSE_EVENT_DONE, "data": response_payload})
    except HTTPException as exc:
        event_queue.put(
            {
                "event": SSE_EVENT_ERROR,
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
                "event": SSE_EVENT_ERROR,
                "data": {
                    "request_id": request_id,
                    "status_code": 500,
                    "detail": str(exc),
                },
            }
        )
    finally:
        event_queue.put({"event": SSE_EVENT_END, "data": {}})


def build_chat_stream_response(
    request: ChatRequest, request_id: str
) -> StreamingResponse:
    """构造 /chat/stream 的 StreamingResponse。

    启动一个 worker 线程跑实际 chat 流程，event_generator 从队列里
    一帧一帧拉出来发给前端；队列空时发 `: ping\\n\\n` 注释帧维持连接。
    """

    event_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
    thread = threading.Thread(
        target=run_chat_stream_worker,
        args=(request, request_id, event_queue),
        daemon=True,
    )
    thread.start()

    def event_generator():
        yield sse_event(
            SSE_EVENT_START,
            {
                "request_id": request_id,
                "session_id": request.session_id.strip(),
            },
        )

        while True:
            try:
                item = event_queue.get(timeout=QUEUE_POLL_TIMEOUT_SECONDS)
            except queue.Empty:
                # SSE 注释行作为心跳，避免中间网关断开空闲连接。
                yield ": ping\n\n"
                continue

            if item["event"] == SSE_EVENT_END:
                break
            yield sse_event(item["event"], item["data"])

    return StreamingResponse(event_generator(), media_type="text/event-stream")

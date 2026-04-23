"""HTTP 路由注册。

使用 APIRouter 把业务路由从 app.py 解耦：
- /chat         非流式
- /chat/stream  SSE 流式

真正的执行逻辑在 chat_runner / streaming 里，本文件只做“收请求 → 分请求 id → 转发”。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter

from .chat_runner import build_chat_result
from .schemas import ChatRequest, ChatResponse
from .streaming import build_chat_stream_response

# request_id 用于日志串联。取 uuid4 的前 12 位保持和旧版一致。
REQUEST_ID_LEN = 12

router = APIRouter()


def _new_request_id() -> str:
    return uuid.uuid4().hex[:REQUEST_ID_LEN]


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    request_id = _new_request_id()
    response_payload, _, _ = build_chat_result(request, request_id)
    return ChatResponse(**response_payload)


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    request_id = _new_request_id()
    return build_chat_stream_response(request, request_id)

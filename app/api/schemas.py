"""HTTP 层的 Pydantic schemas。

独立成一个模块是为了：
- 测试/客户端可以只 import 数据契约，不触发 FastAPI 启动
- 字段演进集中在一处，route / chat_runner / streaming 都消费同一份定义
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat 和 POST /chat/stream 的请求体。"""

    session_id: str
    message: str
    debug: bool = False
    # 请求级 history 文件覆盖，主要给 eval 隔离不同 case 用。
    conversation_history_path: str = ""


class DebugPayload(BaseModel):
    """debug=True 时附带的诊断信息；生产请求不返回。"""

    node_timings: dict[str, float] = Field(default_factory=dict)
    nodes: dict[str, Any] = Field(default_factory=dict)
    tracing: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """/chat 非流式响应；/chat/stream 的 done 帧也复用这个结构。"""

    request_id: str
    session_id: str
    answer: str
    routes: list[str] = []
    summary: str = ""
    debug: DebugPayload | None = None

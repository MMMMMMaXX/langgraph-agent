"""HTTP 层的 Pydantic schemas。

独立成一个模块是为了：
- 测试/客户端可以只 import 数据契约，不触发 FastAPI 启动
- 字段演进集中在一处，route / chat_runner / streaming 都消费同一份定义
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config import CHUNKING_CONFIG
from app.constants.auth import DEFAULT_ROLE
from app.constants.knowledge import (
    RECHUNK_PREVIEW_DEFAULT_SAMPLE_LIMIT,
    RECHUNK_PREVIEW_MAX_CHUNK_SIZE_CHARS,
    RECHUNK_PREVIEW_MAX_MIN_CHUNK_CHARS,
    RECHUNK_PREVIEW_MAX_OVERLAP_CHARS,
    RECHUNK_PREVIEW_MAX_SAMPLE_LIMIT,
    RECHUNK_PREVIEW_MIN_CHUNK_SIZE_CHARS,
    RECHUNK_PREVIEW_MIN_MIN_CHUNK_CHARS,
    RECHUNK_PREVIEW_MIN_OVERLAP_CHARS,
)


class AuthRequest(BaseModel):
    """请求体内的身份透传字段。

    Phase 1 不做 JWT 验签，由上游网关 / BFF 鉴权后透传；后续 middleware
    可在此模型基础上加验签逻辑。
    """

    tenant_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    groups: list[str] = Field(default_factory=list)
    role: str = DEFAULT_ROLE


class ChatRequest(BaseModel):
    """POST /chat 和 POST /chat/stream 的请求体。"""

    session_id: str
    message: str
    debug: bool = False
    # 请求级 history 文件覆盖，主要给 eval 隔离不同 case 用。
    conversation_history_path: str = ""
    # Phase 1：身份注入。None 时走 ALLOW_ANONYMOUS_AUTH 兼容分支。
    auth: AuthRequest | None = None
    # Phase 1：side_effect 工具二次确认 token。第一次请求返回
    # pending_confirmation（内含 token），客户端确认后回填本字段再次请求。
    confirmation_token: str = ""


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
    # side_effect 工具触发了"需要二次确认"时，本字段存签发好的 token 与意图摘要；
    # 客户端下次请求回填 `confirmation_token` 即可真正执行。没有待确认则为 None。
    pending_confirmation: dict[str, Any] | None = None
    # Phase 1：本轮 side_effect 工具执行记录。仅 debug=True 时填充，生产
    # 默认空列表；eval 依赖它聚合 idempotency / tool_safety 指标。
    tool_executions: list[dict[str, Any]] = Field(default_factory=list)
    debug: DebugPayload | None = None


class KnowledgeImportRequest(BaseModel):
    """POST /knowledge/import 的请求体。

    这个入口适合脚本或调试直接传 content；前端文件上传走
    POST /knowledge/import/file。
    """

    content: str
    doc_id: str = ""
    title: str = ""
    source: str = ""
    source_type: str = "txt"
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeImportResponse(BaseModel):
    """文档导入后的索引结果摘要。"""

    doc_id: str
    title: str
    source: str
    source_type: str
    content_hash: str
    content_char_len: int
    chunk_count: int
    indexed_to_sqlite: bool
    indexed_to_chroma: bool


class KnowledgeDocumentListResponse(BaseModel):
    """GET /knowledge/docs 的响应。"""

    documents: list[dict[str, Any]]


class KnowledgeDocumentDetailResponse(BaseModel):
    """GET /knowledge/docs/{doc_id} 的响应。"""

    document: dict[str, Any]


class KnowledgeChunkListResponse(BaseModel):
    """GET /knowledge/docs/{doc_id}/chunks 的响应。"""

    doc_id: str
    total: int
    limit: int
    offset: int
    chunks: list[dict[str, Any]]


class KnowledgeChunkInspectResponse(BaseModel):
    """GET /knowledge/docs/{doc_id}/chunks/inspect 的响应。

    report 是只读诊断报告，用来观察 chunk 长度、章节分布和样例，
    不会触发重新切片或重建索引。
    """

    report: dict[str, Any]


class KnowledgeSearchInspectRequest(BaseModel):
    """POST /knowledge/search/inspect 的请求体。"""

    query: str
    top_k: int = Field(default=8, ge=1, le=50)
    context_preview_chars: int = Field(default=360, ge=0, le=5000)


class KnowledgeSearchInspectResponse(BaseModel):
    """知识库检索解释响应。"""

    report: dict[str, Any]


class KnowledgeRechunkPreviewRequest(BaseModel):
    """POST /knowledge/docs/{doc_id}/rechunk/preview 的请求体。"""

    chunk_size_chars: int = Field(
        default=CHUNKING_CONFIG.chunk_size_chars,
        ge=RECHUNK_PREVIEW_MIN_CHUNK_SIZE_CHARS,
        le=RECHUNK_PREVIEW_MAX_CHUNK_SIZE_CHARS,
    )
    chunk_overlap_chars: int = Field(
        default=CHUNKING_CONFIG.chunk_overlap_chars,
        ge=RECHUNK_PREVIEW_MIN_OVERLAP_CHARS,
        le=RECHUNK_PREVIEW_MAX_OVERLAP_CHARS,
    )
    min_chunk_chars: int = Field(
        default=CHUNKING_CONFIG.min_chunk_chars,
        ge=RECHUNK_PREVIEW_MIN_MIN_CHUNK_CHARS,
        le=RECHUNK_PREVIEW_MAX_MIN_CHUNK_CHARS,
    )
    sample_limit: int = Field(
        default=RECHUNK_PREVIEW_DEFAULT_SAMPLE_LIMIT,
        ge=0,
        le=RECHUNK_PREVIEW_MAX_SAMPLE_LIMIT,
    )


class KnowledgeRechunkPreviewResponse(BaseModel):
    """Rechunk dry-run 响应。"""

    report: dict[str, Any]


class KnowledgeRechunkApplyRequest(KnowledgeRechunkPreviewRequest):
    """POST /knowledge/docs/{doc_id}/rechunk/apply 的请求体。"""


class KnowledgeRechunkApplyResponse(BaseModel):
    """Rechunk apply 响应。"""

    report: dict[str, Any]


class KnowledgeDeleteResponse(BaseModel):
    """DELETE /knowledge/docs/{doc_id} 的响应。"""

    doc_id: str
    deleted: bool
    chunk_count: int
    deleted_from_sqlite: bool
    deleted_from_chroma: bool


class KnowledgeReindexResponse(BaseModel):
    """知识库 Chroma 重建索引的响应。"""

    doc_id: str
    doc_count: int
    chunk_count: int
    reindexed_to_chroma: bool


class KnowledgeHealthResponse(BaseModel):
    """GET /knowledge/health 的响应。"""

    report: dict[str, Any]

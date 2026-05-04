"""HTTP 路由注册。

使用 APIRouter 把业务路由从 app.py 解耦：
- /chat         非流式
- /chat/stream  SSE 流式

真正的执行逻辑在 chat_runner / streaming 里，本文件只做“收请求 → 分请求 id → 转发”。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.knowledge import (
    ChunkQualityThresholds,
    KnowledgeCatalog,
    KnowledgeImportInput,
    RechunkPreviewParams,
    delete_knowledge_document,
    import_knowledge_document,
    inspect_document_chunks,
    inspect_retrieval,
    preview_rechunk_document,
    reindex_all_knowledge_documents,
    reindex_knowledge_document,
)

from .chat_runner import build_chat_result
from .schemas import (
    ChatRequest,
    ChatResponse,
    KnowledgeChunkInspectResponse,
    KnowledgeDeleteResponse,
    KnowledgeDocumentDetailResponse,
    KnowledgeDocumentListResponse,
    KnowledgeImportRequest,
    KnowledgeImportResponse,
    KnowledgeReindexResponse,
    KnowledgeRechunkPreviewRequest,
    KnowledgeRechunkPreviewResponse,
    KnowledgeSearchInspectRequest,
    KnowledgeSearchInspectResponse,
)
from .streaming import build_chat_stream_response

# request_id 用于日志串联。取 uuid4 的前 12 位保持和旧版一致。
REQUEST_ID_LEN = 12
UPLOAD_TEXT_ENCODING = "utf-8-sig"

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


@router.post("/knowledge/import", response_model=KnowledgeImportResponse)
def import_knowledge(request: KnowledgeImportRequest) -> KnowledgeImportResponse:
    """导入单篇知识库文档，并同步写 SQLite FTS5 + Chroma dense index。"""

    try:
        result = import_knowledge_document(
            KnowledgeImportInput(
                content=request.content,
                doc_id=request.doc_id,
                title=request.title,
                source=request.source,
                source_type=request.source_type,
                metadata=request.metadata,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return KnowledgeImportResponse(**result.__dict__)


def _parse_upload_metadata(metadata_json: str) -> dict:
    """解析上传表单中的 metadata_json 字段。"""

    if not metadata_json.strip():
        return {}

    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="metadata_json must be valid JSON object",
        ) from exc

    if not isinstance(metadata, dict):
        raise HTTPException(
            status_code=400,
            detail="metadata_json must be valid JSON object",
        )
    return metadata


def _infer_upload_source_type(filename: str, explicit_source_type: str) -> str:
    """从上传文件名推断 source_type，显式传参优先。"""

    if explicit_source_type.strip():
        return explicit_source_type.strip()

    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix or "txt"


@router.post("/knowledge/import/file", response_model=KnowledgeImportResponse)
async def import_knowledge_file(
    file: Annotated[UploadFile, File(description="txt/md/json 文档文件")],
    doc_id: Annotated[str, Form()] = "",
    title: Annotated[str, Form()] = "",
    source: Annotated[str, Form()] = "",
    source_type: Annotated[str, Form()] = "",
    metadata_json: Annotated[str, Form()] = "",
) -> KnowledgeImportResponse:
    """上传文档文件，并同步写 SQLite FTS5 + Chroma dense index。

    这个入口服务未来前端上传：前端只需要 multipart/form-data 传 file，
    其他字段都是可选表单项。
    """

    raw = await file.read()
    try:
        content = raw.decode(UPLOAD_TEXT_ENCODING)
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="uploaded file must be utf-8 text",
        ) from exc

    metadata = _parse_upload_metadata(metadata_json)
    filename = file.filename or ""
    inferred_source = source.strip() or filename

    try:
        result = import_knowledge_document(
            KnowledgeImportInput(
                content=content,
                doc_id=doc_id,
                title=title,
                source=inferred_source,
                source_type=_infer_upload_source_type(filename, source_type),
                metadata=metadata,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return KnowledgeImportResponse(**result.__dict__)


@router.get("/knowledge/docs", response_model=KnowledgeDocumentListResponse)
def list_knowledge_docs(
    limit: int = Query(default=50, ge=0, le=200),
    offset: int = Query(default=0, ge=0),
) -> KnowledgeDocumentListResponse:
    """查看已导入文档列表。"""

    catalog = KnowledgeCatalog()
    return KnowledgeDocumentListResponse(
        documents=catalog.list_documents(limit=limit, offset=offset)
    )


@router.get("/knowledge/docs/{doc_id}", response_model=KnowledgeDocumentDetailResponse)
def get_knowledge_doc(doc_id: str) -> KnowledgeDocumentDetailResponse:
    """查看单篇文档及 chunk 摘要。"""

    catalog = KnowledgeCatalog()
    document = catalog.get_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return KnowledgeDocumentDetailResponse(document=document)


def _build_chunk_quality_thresholds(
    *,
    short_chars: int | None,
    long_chars: int | None,
) -> ChunkQualityThresholds | None:
    """把查询参数转成 chunk 质量阈值；不传时使用 chunking 默认配置。"""

    if short_chars is None and long_chars is None:
        return None

    defaults = ChunkQualityThresholds()
    return ChunkQualityThresholds(
        short_chars=short_chars or defaults.short_chars,
        long_chars=long_chars or defaults.long_chars,
    )


@router.get(
    "/knowledge/docs/{doc_id}/chunks/inspect",
    response_model=KnowledgeChunkInspectResponse,
)
def inspect_knowledge_doc_chunks(
    doc_id: str,
    sample_limit: int = Query(default=5, ge=0, le=50),
    short_chars: int | None = Query(default=None, ge=0),
    long_chars: int | None = Query(default=None, ge=0),
) -> KnowledgeChunkInspectResponse:
    """查看单篇文档的 chunk 质量统计。

    这是诊断接口，只读 SQLite catalog，不修改文档、不重建 Chroma。
    """

    catalog = KnowledgeCatalog()
    if catalog.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="document not found")

    report = inspect_document_chunks(
        doc_id,
        catalog=catalog,
        sample_limit=sample_limit,
        thresholds=_build_chunk_quality_thresholds(
            short_chars=short_chars,
            long_chars=long_chars,
        ),
    )
    return KnowledgeChunkInspectResponse(report=asdict(report))


@router.post(
    "/knowledge/search/inspect",
    response_model=KnowledgeSearchInspectResponse,
)
def inspect_knowledge_search(
    request: KnowledgeSearchInspectRequest,
) -> KnowledgeSearchInspectResponse:
    """解释一个 query 的知识库检索链路。"""

    try:
        report = inspect_retrieval(
            request.query,
            top_k=request.top_k,
            context_preview_chars=request.context_preview_chars,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return KnowledgeSearchInspectResponse(report=asdict(report))


@router.get(
    "/knowledge/search/inspect",
    response_model=KnowledgeSearchInspectResponse,
)
def inspect_knowledge_search_get(
    query: str = Query(..., min_length=1),
    top_k: int = Query(default=8, ge=1, le=50),
    context_preview_chars: int = Query(default=360, ge=0, le=5000),
) -> KnowledgeSearchInspectResponse:
    """GET 版本，方便浏览器和 curl 快速调试。"""

    try:
        report = inspect_retrieval(
            query,
            top_k=top_k,
            context_preview_chars=context_preview_chars,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return KnowledgeSearchInspectResponse(report=asdict(report))


@router.post(
    "/knowledge/docs/{doc_id}/rechunk/preview",
    response_model=KnowledgeRechunkPreviewResponse,
)
def preview_knowledge_doc_rechunk(
    doc_id: str,
    request: KnowledgeRechunkPreviewRequest,
) -> KnowledgeRechunkPreviewResponse:
    """预览重新切片结果；只读 dry-run，不写 SQLite/Chroma。"""

    try:
        report = preview_rechunk_document(
            doc_id,
            params=RechunkPreviewParams(
                chunk_size_chars=request.chunk_size_chars,
                chunk_overlap_chars=request.chunk_overlap_chars,
                min_chunk_chars=request.min_chunk_chars,
                sample_limit=request.sample_limit,
            ),
        )
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if detail == "document not found" else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return KnowledgeRechunkPreviewResponse(report=asdict(report))


@router.delete("/knowledge/docs/{doc_id}", response_model=KnowledgeDeleteResponse)
def delete_knowledge_doc(doc_id: str) -> KnowledgeDeleteResponse:
    """删除单篇文档，同时清理 SQLite/FTS5 和 Chroma。"""

    result = delete_knowledge_document(doc_id)
    if not result.deleted:
        raise HTTPException(status_code=404, detail="document not found")
    return KnowledgeDeleteResponse(**result.__dict__)


@router.post(
    "/knowledge/docs/{doc_id}/reindex",
    response_model=KnowledgeReindexResponse,
)
def reindex_knowledge_doc(doc_id: str) -> KnowledgeReindexResponse:
    """从 SQLite catalog 重建单篇文档的 Chroma dense index。"""

    result = reindex_knowledge_document(doc_id)
    if not result.reindexed_to_chroma:
        raise HTTPException(status_code=404, detail="document not found")
    return KnowledgeReindexResponse(**result.__dict__)


@router.post("/knowledge/reindex", response_model=KnowledgeReindexResponse)
def reindex_knowledge_all() -> KnowledgeReindexResponse:
    """从 SQLite catalog 全量重建 Chroma docs collection。"""

    result = reindex_all_knowledge_documents()
    return KnowledgeReindexResponse(**result.__dict__)

from app.config import VECTOR_STORE_CONFIG
from app.constants.model_profiles import (
    PROFILE_MEMORY_EMBEDDING,
    PROFILE_QUERY_EMBEDDING,
)
from app.constants.tags import CITY_TAGS, LITERAL_MATCH_WEIGHT, TAG_WEIGHTS, TOPIC_TAGS
from app.retrieval.embedder import get_embedding
from app.utils.logger import now_timestamp_s
from app.utils.memory_key import (
    MEMORY_TYPE_FACT,
    build_memory_key,
    classify_memory_type,
)
from app.utils.tags import extract_tags
from app.vector_store import ChromaVectorStore

MEMORY_SCHEMA_VERSION = 1
MEMORY_SOURCE_CHAT_ROUND = "chat_round"
MEMORY_SOURCE_CONVERSATION = "conversation"
DEFAULT_MEMORY_CONFIDENCE = 0.8

# Chroma get() 分页大小，避免一次性拉回大量 memory 记录。
MEMORY_GET_PAGE_SIZE = 100

# 最近 memory 至少扫描的尾部窗口大小，保证小 session 下仍有足够候选排序。
RECENT_MEMORY_MIN_SCAN_LIMIT = 50

# 最近 memory 最多扫描的尾部窗口大小，防止长 session 下 get() 拉取过多记录。
RECENT_MEMORY_MAX_SCAN_LIMIT = 500

# 最近 memory 扫描窗口相对目标 limit 的放大倍数。
RECENT_MEMORY_SCAN_MULTIPLIER = 20


def _serialize_tags(tags: list[str]) -> str:
    """把 tags 列表转成 Chroma metadata 可安全存储的字符串。

    Chroma metadata 更适合保存标量值，而不是复杂的 list 结构。
    这里用逗号拼接的轻量形式保存，读取时再反解回来，
    能兼顾：
    - 元数据兼容性
    - 代码实现简单
    - 对现有 tags 逻辑侵入最小
    """

    return ",".join(tags)


def _deserialize_tags(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(",") if item]


def _build_memory_record(
    *,
    document_id: str,
    content: str,
    metadata: dict,
) -> dict:
    """把 Chroma 的 document + metadata 还原成项目内部 memory 记录结构。"""

    return {
        "id": document_id,
        "content": content,
        "source": metadata.get("source", "conversation"),
        "session_id": metadata.get("session_id", "default"),
        "timestamp": metadata.get("timestamp", 0),
        "tags": _deserialize_tags(metadata.get("tags", "")),
        "memory_key": metadata.get("memory_key", ""),
        "memory_type": metadata.get("memory_type", MEMORY_TYPE_FACT),
        "confidence": float(metadata.get("confidence", DEFAULT_MEMORY_CONFIDENCE)),
        "schema_version": int(metadata.get("schema_version", 0)),
        "source_route": metadata.get("source_route", ""),
        "rewritten_query": metadata.get("rewritten_query", ""),
        "is_active": bool(metadata.get("is_active", 1)),
    }


def _flatten_get_result(result: dict) -> list[dict]:
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    records: list[dict] = []

    for index, document_id in enumerate(ids):
        content = documents[index] if index < len(documents) else ""
        metadata = metadatas[index] if index < len(metadatas) else {}
        records.append(
            _build_memory_record(
                document_id=document_id,
                content=content,
                metadata=metadata,
            )
        )

    return records


def _flatten_query_result(result: dict) -> list[dict]:
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    hits: list[dict] = []

    for index, document_id in enumerate(ids):
        content = documents[index] if index < len(documents) else ""
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        record = _build_memory_record(
            document_id=document_id,
            content=content,
            metadata=metadata,
        )
        record["semantic_score"] = (
            max(0.0, 1.0 - float(distance)) if distance is not None else 0.0
        )
        hits.append(record)

    return hits


def _build_memory_document_id(session_id: str, memory_key: str) -> str:
    """构造 memory 文档主键。

    当前 memory 不做 chunk，所以直接用：
    `session_id + memory_key`
    表达一条逻辑记忆的唯一身份。

    这样同一 session 下相同 memory_key 的写入天然就是 upsert，
    可以复用原来 JSON 版“覆盖旧记录”的语义。
    """

    return f"{session_id}::memory::{memory_key}"


def build_memory_metadata(
    *,
    session_id: str,
    source: str,
    timestamp: float,
    tags: list[str],
    memory_key: str,
    memory_type: str,
    rewritten_query: str = "",
    confidence: float = DEFAULT_MEMORY_CONFIDENCE,
    source_route: str = "",
    is_active: bool = True,
) -> dict:
    """构造 Chroma memory 的标准 metadata。

    Chroma metadata 适合存标量，因此 tags 用逗号字符串保存。
    这里集中声明 schema，避免在线写入和迁移脚本各拼一套字段。
    """

    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "session_id": session_id,
        "source": source,
        "source_route": source_route,
        "timestamp": timestamp,
        "tags": _serialize_tags(tags),
        "memory_key": memory_key,
        "memory_type": memory_type,
        "confidence": confidence,
        "rewritten_query": rewritten_query,
        "is_active": 1 if is_active else 0,
    }


def _get_store() -> ChromaVectorStore:
    return ChromaVectorStore()


def add_memory_item(
    content: str,
    source: str = MEMORY_SOURCE_CONVERSATION,
    rewritten_query: str = "",
    session_id: str = "default",
    source_route: str = "",
    confidence: float = DEFAULT_MEMORY_CONFIDENCE,
    tags: list[str] | None = None,
    memory_key: str = "",
    memory_type: str = "",
) -> None:
    tags = tags if tags is not None else extract_tags(rewritten_query or content)
    embedding = get_embedding(content, profile=PROFILE_MEMORY_EMBEDDING)
    memory_key = memory_key or build_memory_key(rewritten_query or content, tags)
    memory_type = memory_type or classify_memory_type(rewritten_query)
    now = now_timestamp_s()
    document_id = _build_memory_document_id(session_id, memory_key)
    store = _get_store()

    # memory 暂时不做 chunk，所以一条回答就是一条 document。
    # 这里直接按 document_id upsert，即可保留旧逻辑里的“同 key 覆盖更新”。
    store.upsert(
        collection_name=VECTOR_STORE_CONFIG.memory_collection_name,
        ids=[document_id],
        documents=[content],
        embeddings=[embedding],
        metadatas=[
            build_memory_metadata(
                session_id=session_id,
                source=source,
                source_route=source_route,
                timestamp=now,
                tags=tags,
                memory_key=memory_key,
                memory_type=memory_type,
                rewritten_query=rewritten_query,
                confidence=confidence,
                is_active=True,
            )
        ],
    )


def keyword_score_for_memory(query: str, item: dict) -> float:
    score = 0.0
    query_tags = extract_tags(query)
    item_tags = item.get("tags", [])
    content = item.get("content", "")

    for tag in query_tags:
        if tag in item_tags or tag in content:
            if tag in CITY_TAGS or tag in TOPIC_TAGS:
                score += TAG_WEIGHTS.get(tag, 0.0)

    if query.strip() and query.strip() in content:
        score += LITERAL_MATCH_WEIGHT

    return score


def normalize_memory_keyword_scores(hits: list[dict]) -> list[dict]:
    if not hits:
        return hits

    max_score = max(h["keyword_score"] for h in hits)
    if max_score <= 0:
        for h in hits:
            h["keyword_score_norm"] = 0.0
        return hits

    for h in hits:
        h["keyword_score_norm"] = h["keyword_score"] / max_score
    return hits


def search_memory(
    query: str,
    top_k: int = 5,
    alpha: float = 0.7,
    beta: float = 0.3,
    session_id: str | None = None,
) -> list[dict]:
    if not query or not query.strip():
        return []

    query_emb = get_embedding(query, profile=PROFILE_QUERY_EMBEDDING)
    store = _get_store()
    # memory 检索时限制当前 session_id，避免跨 session 混入
    where = {"session_id": session_id} if session_id is not None else None

    # 先扩大候选召回，再在项目内部用 keyword/hybrid 重排，
    # 保持和 doc 侧一致的迁移策略。
    raw_result = store.query(
        collection_name=VECTOR_STORE_CONFIG.memory_collection_name,
        query_embedding=query_emb,
        top_k=max(top_k * 4, top_k),
        where=where,
    )
    hits = _flatten_query_result(raw_result)

    hits = [item for item in hits if item.get("is_active", True)]

    for hit in hits:
        hit["keyword_score"] = keyword_score_for_memory(query, hit)

    hits = normalize_memory_keyword_scores(hits)

    for h in hits:
        h["score"] = alpha * h["semantic_score"] + beta * h["keyword_score_norm"]

    hits.sort(key=lambda x: (x["score"], x["timestamp"]), reverse=True)
    return hits[:top_k]


def get_recent_memory(
    session_id: str,
    limit: int = 5,
) -> list[dict]:
    """读取当前 session 最近写入的 memory。

    这个接口专门服务“总结刚刚/刚才的问题”：
    - 不走语义向量召回
    - 只按 session_id 过滤
    - 再按 timestamp 取最近几条

    这样可以避免“总结刚才”被当前 session 中很早以前但语义相近的历史记录污染。
    """

    if limit <= 0:
        return []

    store = _get_store()
    collection_name = VECTOR_STORE_CONFIG.memory_collection_name
    total_count = store.count(collection_name=collection_name)
    scan_limit = min(
        max(limit * RECENT_MEMORY_SCAN_MULTIPLIER, RECENT_MEMORY_MIN_SCAN_LIMIT),
        RECENT_MEMORY_MAX_SCAN_LIMIT,
    )
    start_offset = max(total_count - scan_limit, 0)
    records: list[dict] = []

    for offset in range(start_offset, total_count, MEMORY_GET_PAGE_SIZE):
        result = store.get(
            collection_name=collection_name,
            where={"session_id": session_id},
            limit=MEMORY_GET_PAGE_SIZE,
            offset=offset,
        )
        page_records = [
            item for item in _flatten_get_result(result) if item.get("is_active", True)
        ]
        records.extend(page_records)

    records.sort(key=lambda item: item.get("timestamp", 0), reverse=True)
    return sorted(records[:limit], key=lambda item: item.get("timestamp", 0))


def build_global_memory_index(session_id: str | None = None):
    store = _get_store()
    where = {"session_id": session_id} if session_id is not None else None
    collection_name = VECTOR_STORE_CONFIG.memory_collection_name
    data: list[dict] = []
    offset = 0

    while True:
        result = store.get(
            collection_name=collection_name,
            where=where,
            limit=MEMORY_GET_PAGE_SIZE,
            offset=offset,
        )
        page_data = _flatten_get_result(result)
        if not page_data:
            break

        data.extend(page_data)
        if len(page_data) < MEMORY_GET_PAGE_SIZE:
            break
        offset += MEMORY_GET_PAGE_SIZE

    cities = set()
    topics = set()

    for item in data:
        if session_id is not None and item.get("session_id", "default") != session_id:
            continue
        tags = item.get("tags", [])

        for t in tags:
            if t in CITY_TAGS:
                cities.add(t)
            if t in TOPIC_TAGS:
                topics.add(t)

    return {
        "cities": sorted(cities),
        "topics": sorted(topics),
    }

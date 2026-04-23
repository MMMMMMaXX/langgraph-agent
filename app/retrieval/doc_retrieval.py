from app.constants.tags import LITERAL_MATCH_WEIGHT, TAG_MATCH_TERMS, TAG_WEIGHTS
from app.config import RAG_CONFIG, VECTOR_STORE_CONFIG
from app.constants.model_profiles import PROFILE_QUERY_EMBEDDING
from app.retrieval.embedder import get_embedding
from app.utils.tags import extract_tags
from app.vector_store import ChromaVectorStore

DEFAULT_HYBRID_ALPHA = 0.65
DEFAULT_HYBRID_BETA = 0.35
DOC_CANDIDATE_MULTIPLIER = 4
DEFAULT_CHUNK_INDEX = 0
RETRIEVAL_SOURCE_DENSE = "dense"
RETRIEVAL_SOURCE_KEYWORD = "keyword"


def normalize_keyword_text(text: str) -> str:
    """归一化关键词匹配文本，避免英文大小写导致 keyword 分丢失。"""

    return text.strip().lower()


def keyword_score(query: str, content: str) -> float:
    """计算 query 和单个 chunk 的轻量关键词匹配分。

    这个分数不是最终排序分，只负责补足向量召回对专有名词、英文缩写、
    中文短词不够稳定的问题。后续会在 `normalize_keyword_scores()` 中归一化，
    再交给 hybrid ranker 和 semantic_score 融合。
    """

    score = 0.0
    query_tags = extract_tags(query)
    normalized_content = normalize_keyword_text(content)
    normalized_query = normalize_keyword_text(query)

    for tag in query_tags:
        match_terms = TAG_MATCH_TERMS.get(tag, (tag,))
        normalized_terms = [normalize_keyword_text(term) for term in match_terms]
        if any(term and term in normalized_content for term in normalized_terms):
            score += TAG_WEIGHTS.get(tag, 0.0)

    # 字面包含作为补充，适合非常短且高度确定的查询。
    if normalized_query and normalized_query in normalized_content:
        score += LITERAL_MATCH_WEIGHT

    return score


def normalize_keyword_scores(hits: list[dict]) -> list[dict]:
    """把 keyword_score 归一化到 0~1，方便和 semantic_score 融合。"""

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


def distance_to_semantic_score(distance: float | None) -> float:
    """把 Chroma 的 cosine distance 转成项目内部统一使用的 semantic_score。

    当前 collection 使用 `hnsw:space=cosine`：
    - distance 越小越相似
    - 这里把它近似映射成 0~1 的相似度分数，方便和既有 keyword/hybrid 逻辑复用
    """

    if distance is None:
        return 0.0
    return max(0.0, 1.0 - float(distance))


def flatten_chroma_query_result(result: dict) -> list[dict]:
    """把 Chroma query 的嵌套返回结构拍平为项目内部 hit 列表。"""

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]

    hits: list[dict] = []
    for index, content in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        item_id = ids[index] if index < len(ids) else ""

        hits.append(
            {
                "id": item_id,
                "content": content,
                "semantic_score": distance_to_semantic_score(distance),
                "distance": distance,
                "doc_id": str(metadata.get("doc_id", "")),
                "doc_title": str(metadata.get("doc_title", "")),
                "source": str(metadata.get("source", "")),
                "chunk_index": metadata.get("chunk_index", DEFAULT_CHUNK_INDEX),
                "start_char": metadata.get("start_char", 0),
                "end_char": metadata.get("end_char", 0),
                "chunk_char_len": metadata.get("chunk_char_len", len(content)),
            }
        )

    return hits


def build_doc_hit(
    *,
    item_id: str,
    content: str,
    metadata: dict,
    semantic_score: float = 0.0,
    distance: float | None = None,
) -> dict:
    """把 Chroma 记录转换成项目内部统一的 doc hit 结构。"""

    return {
        "id": item_id,
        "content": content,
        "semantic_score": semantic_score,
        "distance": distance,
        "doc_id": str(metadata.get("doc_id", "")),
        "doc_title": str(metadata.get("doc_title", "")),
        "source": str(metadata.get("source", "")),
        "chunk_index": metadata.get("chunk_index", DEFAULT_CHUNK_INDEX),
        "start_char": metadata.get("start_char", 0),
        "end_char": metadata.get("end_char", 0),
        "chunk_char_len": metadata.get("chunk_char_len", len(content)),
    }


def flatten_chroma_get_result(result: dict) -> list[dict]:
    """把 Chroma get 的扁平返回结构转换成项目内部 hit 列表。"""

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    ids = result.get("ids") or []

    hits: list[dict] = []
    for index, content in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        item_id = ids[index] if index < len(ids) else ""
        hits.append(
            build_doc_hit(
                item_id=item_id,
                content=content,
                metadata=metadata,
            )
        )

    return hits


def merge_doc_hits(hit_groups: list[list[dict]]) -> list[dict]:
    """按 chunk id 合并多路召回结果，并保留召回来源。

    Dense 和 keyword 可能命中同一个 chunk。合并时以 id 去重，并保留更高的
    semantic_score / keyword_score，避免同一内容重复进入后续 ranker。
    """

    merged: dict[str, dict] = {}

    for hits in hit_groups:
        for hit in hits:
            item_id = hit.get("id", "")
            if not item_id:
                continue

            retrieval_source = hit.get("retrieval_source", "")
            existing = merged.get(item_id)
            if existing is None:
                item = dict(hit)
                item["retrieval_sources"] = (
                    [retrieval_source] if retrieval_source else []
                )
                merged[item_id] = item
                continue

            if (
                retrieval_source
                and retrieval_source not in existing["retrieval_sources"]
            ):
                existing["retrieval_sources"].append(retrieval_source)

            if hit.get("semantic_score", 0.0) > existing.get("semantic_score", 0.0):
                existing["semantic_score"] = hit.get("semantic_score", 0.0)
                existing["distance"] = hit.get("distance")

            if hit.get("keyword_score", 0.0) > existing.get("keyword_score", 0.0):
                existing["keyword_score"] = hit.get("keyword_score", 0.0)

    return list(merged.values())


def dense_retrieve_docs(query: str, top_k: int) -> list[dict]:
    """用 query embedding 从 Chroma docs collection 召回候选 chunk。"""

    query_emb = get_embedding(query, profile=PROFILE_QUERY_EMBEDDING)
    store = ChromaVectorStore()
    raw_result = store.query(
        collection_name=VECTOR_STORE_CONFIG.doc_collection_name,
        query_embedding=query_emb,
        top_k=top_k,
    )
    hits = flatten_chroma_query_result(raw_result)
    for hit in hits:
        hit["retrieval_source"] = RETRIEVAL_SOURCE_DENSE
    return hits


def keyword_retrieve_docs(query: str, top_k: int) -> list[dict]:
    """从 Chroma docs collection 做轻量关键词召回。

    当前实现直接扫描 collection 中的 chunk，适合我们现阶段的小规模学习项目。
    后续如果文档量变大，可以把这个函数替换成 BM25/倒排索引，而不影响
    search_docs 的主流程。
    """

    store = ChromaVectorStore()
    raw_result = store.get(collection_name=VECTOR_STORE_CONFIG.doc_collection_name)
    hits = flatten_chroma_get_result(raw_result)

    keyword_hits = []
    for hit in hits:
        score = keyword_score(query, hit["content"])
        if score <= 0:
            continue
        hit["keyword_score"] = score
        hit["retrieval_source"] = RETRIEVAL_SOURCE_KEYWORD
        keyword_hits.append(hit)

    keyword_hits.sort(
        key=lambda item: (
            item.get("keyword_score", 0.0),
            -item.get("chunk_index", DEFAULT_CHUNK_INDEX),
        ),
        reverse=True,
    )
    return keyword_hits[:top_k]


def apply_keyword_scores(query: str, hits: list[dict]) -> list[dict]:
    """为候选 chunk 补充 keyword_score 和 keyword_score_norm。"""

    for hit in hits:
        hit["keyword_score"] = keyword_score(query, hit["content"])

    return normalize_keyword_scores(hits)


def rank_hybrid(
    hits: list[dict],
    *,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    beta: float = DEFAULT_HYBRID_BETA,
) -> list[dict]:
    """按 semantic + keyword 的 hybrid 分数排序。"""

    for hit in hits:
        hit["score"] = alpha * hit["semantic_score"] + beta * hit["keyword_score_norm"]

    hits.sort(
        key=lambda item: (
            item["score"],
            item["semantic_score"],
            -item.get("chunk_index", DEFAULT_CHUNK_INDEX),
        ),
        reverse=True,
    )
    return hits


def search_docs(
    query: str,
    top_k: int = RAG_CONFIG.doc_top_k,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    beta: float = DEFAULT_HYBRID_BETA,
) -> list[dict]:
    """从 Chroma docs collection 召回 chunk，并沿用现有 hybrid 打分逻辑。

    当前检索管线保持行为不变，但把职责拆成稳定步骤：
    1. dense_retrieve_docs：向量召回候选 chunk
    2. apply_keyword_scores：补充关键词信号
    3. rank_hybrid：融合 semantic / keyword 并排序

    后续要接 BM25、多路召回、相邻 chunk 合并时，可以分别接在这些边界上。
    """

    if not query or not query.strip():
        return []

    # 先多取一些候选，再在项目内做 keyword + hybrid 重排，
    # 否则 Chroma 只返回 top_k 时，keyword 信号的发挥空间会太小。
    candidate_top_k = max(top_k * DOC_CANDIDATE_MULTIPLIER, top_k)
    dense_hits = dense_retrieve_docs(query, candidate_top_k)
    keyword_hits = keyword_retrieve_docs(query, candidate_top_k)
    hits = merge_doc_hits([dense_hits, keyword_hits])
    hits = apply_keyword_scores(query, hits)
    hits = rank_hybrid(hits, alpha=alpha, beta=beta)
    return hits[:top_k]

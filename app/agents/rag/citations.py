"""RAG 文档引用构造。

这里把“给 LLM 看的引用文本”和“给 API/debug/eval 用的结构化 citation”
放在同一处生成，避免 prompt、debug、eval 各自拼一套来源字段后逐渐漂移。
"""

from app.agents.rag.constants import DOC_HIT_DEBUG_TEXT_PREVIEW_CHARS
from app.utils.logger import preview


def get_chunk_identifier(doc_hit: dict) -> str:
    """返回最适合展示的 chunk 标识。"""

    merged_chunk_ids = doc_hit.get("merged_chunk_ids") or []
    if merged_chunk_ids:
        return "+".join(str(item) for item in merged_chunk_ids)

    return str(doc_hit.get("id", ""))


def build_citation(index: int, doc_hit: dict) -> dict:
    """把一个最终上下文片段转换成 citation 元数据。"""

    return {
        "index": index,
        "ref": f"[{index}]",
        "id": doc_hit.get("id", ""),
        "doc_id": str(doc_hit.get("doc_id", "")),
        "chunk_id": get_chunk_identifier(doc_hit),
        "doc_title": doc_hit.get("doc_title", ""),
        "source": doc_hit.get("source", ""),
        "section_title": doc_hit.get("section_title", ""),
        "chunk_index": doc_hit.get("chunk_index", 0),
        "merged_chunk_ids": doc_hit.get("merged_chunk_ids", []),
        "merged_chunk_indexes": doc_hit.get("merged_chunk_indexes", []),
        "score": round(float(doc_hit.get("score", 0.0) or 0.0), 4),
        "semantic_score": round(float(doc_hit.get("semantic_score", 0.0) or 0.0), 4),
        "keyword_score_norm": round(
            float(doc_hit.get("keyword_score_norm", 0.0) or 0.0),
            4,
        ),
        "preview": preview(
            doc_hit.get("content", ""),
            DOC_HIT_DEBUG_TEXT_PREVIEW_CHARS,
        ),
    }


def build_citations(doc_hits: list[dict], max_blocks: int) -> list[dict]:
    """为送入回答模型的文档片段生成稳定编号。"""

    citations = []
    citation_index = 1
    for doc_hit in doc_hits[:max_blocks]:
        content = doc_hit.get("content", "").strip()
        if not content:
            continue
        citations.append(build_citation(citation_index, doc_hit))
        citation_index += 1
    return citations


def format_source_label(citation: dict) -> str:
    """生成适合放进 prompt 的短来源标签。"""

    parts = []
    if citation.get("doc_title"):
        parts.append(f"标题：{citation['doc_title']}")
    if citation.get("source"):
        parts.append(f"来源：{citation['source']}")
    if citation.get("section_title"):
        parts.append(f"章节：{citation['section_title']}")
    if citation.get("chunk_index") not in (None, ""):
        parts.append(f"chunk：{citation['chunk_index']}")

    return "；".join(parts)


def format_cited_doc_block(doc_hit: dict, citation: dict, max_chars: int) -> str:
    """生成带引用编号的单个上下文块。"""

    content = doc_hit.get("content", "").strip()
    source_label = format_source_label(citation)
    header = citation["ref"]
    if source_label:
        header += f" {source_label}"

    return f"{header}\n{content[:max_chars]}"

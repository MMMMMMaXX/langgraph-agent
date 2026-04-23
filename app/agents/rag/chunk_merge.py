"""RAG 相邻 chunk 合并逻辑。

检索阶段仍然以 chunk 为基本单位；生成阶段可以把同一文档的连续 chunk
合并成更自然的上下文，减少模型看到“半句话开头/半句话结尾”的概率。
"""

from app.agents.rag.constants import MERGED_CHUNK_SEPARATOR


def merge_adjacent_doc_hits(hits: list[dict]) -> list[dict]:
    """合并同一文档中连续命中的 chunk，构造更自然的生成上下文。

    检索和 debug 仍然保留原始 chunk 列表；这里仅影响最终送入生成模型的
    doc_hits。这样模型看到的是连续段落，而不是被 chunk 边界切碎的片段。
    """

    if not hits:
        return []

    ordered_hits = sorted(
        hits,
        key=lambda hit: (
            hit.get("doc_id", ""),
            hit.get("chunk_index", 0),
            hit.get("start_char", 0),
        ),
    )
    merged_hits: list[dict] = []
    current_group: list[dict] = []

    def flush_group() -> None:
        if not current_group:
            return

        base = dict(current_group[0])
        chunk_ids = [hit.get("id", "") for hit in current_group]
        chunk_indexes = [hit.get("chunk_index", 0) for hit in current_group]
        retrieval_sources = sorted(
            {
                source
                for hit in current_group
                for source in (hit.get("retrieval_sources") or [])
            }
        )
        base["id"] = "+".join(chunk_ids)
        base["content"] = MERGED_CHUNK_SEPARATOR.join(
            hit.get("content", "").strip()
            for hit in current_group
            if hit.get("content", "").strip()
        )
        base["start_char"] = min(hit.get("start_char", 0) for hit in current_group)
        base["end_char"] = max(hit.get("end_char", 0) for hit in current_group)
        base["chunk_char_len"] = len(base["content"])
        base["chunk_index"] = min(chunk_indexes)
        base["score"] = max(hit.get("score", 0.0) for hit in current_group)
        base["semantic_score"] = max(
            hit.get("semantic_score", 0.0) for hit in current_group
        )
        base["keyword_score_norm"] = max(
            hit.get("keyword_score_norm", 0.0) for hit in current_group
        )
        base["retrieval_sources"] = retrieval_sources
        base["merged_chunk_ids"] = chunk_ids
        base["merged_chunk_indexes"] = chunk_indexes
        merged_hits.append(base)

    for hit in ordered_hits:
        if not current_group:
            current_group.append(hit)
            continue

        previous = current_group[-1]
        is_adjacent = (
            hit.get("doc_id") == previous.get("doc_id")
            and hit.get("chunk_index", 0) == previous.get("chunk_index", 0) + 1
        )
        if is_adjacent:
            current_group.append(hit)
            continue

        flush_group()
        current_group.clear()
        current_group.append(hit)

    flush_group()

    # 合并后按原始最高分恢复排序，避免上下文顺序因为 doc_id 排序被改变。
    merged_hits.sort(
        key=lambda hit: (
            hit.get("score", 0.0),
            hit.get("semantic_score", 0.0),
            -hit.get("chunk_index", 0),
        ),
        reverse=True,
    )
    return merged_hits


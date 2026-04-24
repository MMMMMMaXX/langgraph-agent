"""RAG 生成上下文压缩。"""

from app.agents.rag.types import RagContext
from app.config import RAG_CONFIG


def compress_doc_context(doc_hits: list[dict]) -> str:
    """按默认 RAG 配置压缩文档上下文。"""

    return compress_doc_context_with_limit(
        doc_hits,
        RAG_CONFIG.max_doc_context_chars,
    )


def compress_doc_context_with_limit(doc_hits: list[dict], max_chars: int) -> str:
    """按字符上限截断文档上下文，避免生成阶段输入过长。

    生成阶段通常是最贵的一步，这里先用轻量截断压缩上下文，
    避免把整段长文档原样塞给模型，导致生成耗时和 token 成本过高。
    """

    selected_blocks = []

    for doc in doc_hits[: RAG_CONFIG.max_doc_context_blocks]:
        content = doc.get("content", "").strip()
        if not content:
            continue
        selected_blocks.append(content[:max_chars])

    return "\n".join(selected_blocks)


def build_rag_context(
    *,
    doc_hits: list[dict],
    memory_hits: list[dict],
    doc_context_chars: int,
) -> RagContext:
    """构建 RAG 生成阶段需要的完整上下文。

    context 是 doc/memory 的完整拼接版本；doc_context 和 memory_context
    分别保留给不同回答分支使用，避免 node.py 里散落字符串拼接。
    """

    doc_context = compress_doc_context_with_limit(doc_hits, doc_context_chars)
    memory_context = "\n".join(
        memory_hit["content"] for memory_hit in memory_hits if memory_hit["score"] >= 1
    )

    context = ""
    if doc_hits:
        context += "知识库资料：\n"
        context += doc_context + "\n"

    if memory_hits:
        context += "\n历史相关记录：\n"
        for memory_hit in memory_hits:
            context += memory_hit["content"] + "\n"

    return RagContext(
        context=context,
        doc_context=doc_context,
        memory_context=memory_context,
    )

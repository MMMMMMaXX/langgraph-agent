"""RAG 生成上下文压缩。

这里的 compression 是规则版、低成本的第一层压缩：
- citation 仍保持 block 级别，引用编号不变；
- 只压缩送入 LLM 的正文，减少无关句子和 token 浪费；
- 不调用 LLM，避免压缩阶段引入额外成本和不稳定性。
"""

from __future__ import annotations

import re

from app.agents.rag.citations import build_citations, format_cited_doc_block
from app.agents.rag.constants import (
    CONTEXT_COMPRESSION_DEFINITION_SIGNALS,
    CONTEXT_COMPRESSION_MAX_SENTENCES_PER_BLOCK,
    CONTEXT_COMPRESSION_MIN_BLOCK_CHARS,
    MEMORY_COMPRESSION_MAX_BLOCK_CHARS,
    MEMORY_COMPRESSION_MAX_HITS,
    MEMORY_COMPRESSION_MAX_TOTAL_CHARS,
    QUERY_TYPE_DEFINITION,
)
from app.agents.rag.types import RagContext
from app.config import RAG_CONFIG
from app.retrieval.lexical.tokenizer import lexical_terms


SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


def _split_sentences(text: str) -> list[str]:
    """按中文/英文标点和换行切出候选句，保留原句顺序。"""

    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]


def _normalize_terms(query: str) -> list[str]:
    """提取 query 关键词，统一大小写并去掉过短噪声。"""

    terms = []
    seen: set[str] = set()
    for term in lexical_terms(query):
        normalized = term.lower().strip()
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
    return terms


def _score_sentence(
    *,
    sentence: str,
    sentence_index: int,
    query_terms: list[str],
    query_type: str,
) -> float:
    """给候选句打相关性分。

    分数来自三类轻量信号：
    1. query term 覆盖度，确保保留和问题最相关的句子；
    2. 定义类信号词，例如“是/指/用于/作用”，帮助“X 是什么”保留定义句；
    3. 位置轻微加权，很多文档首句就是定义或摘要。
    """

    lowered = sentence.lower()
    score = 0.0

    for term in query_terms:
        if term and term in lowered:
            score += 2.0

    if query_type == QUERY_TYPE_DEFINITION:
        score += sum(
            0.5
            for signal in CONTEXT_COMPRESSION_DEFINITION_SIGNALS
            if signal in sentence
        )

    if sentence_index == 0:
        score += 0.5

    return score


def _compress_block_content(
    *,
    content: str,
    query: str,
    query_type: str,
    max_chars: int,
) -> tuple[str, dict]:
    """压缩单个 citation block 的正文，并返回压缩统计。"""

    original = content.strip()
    if not original:
        return "", {
            "before_chars": 0,
            "after_chars": 0,
            "sentence_count": 0,
            "selected_sentence_count": 0,
        }

    sentences = _split_sentences(original)
    if not sentences:
        compressed = original[:max_chars]
        return compressed, {
            "before_chars": len(original),
            "after_chars": len(compressed),
            "sentence_count": 0,
            "selected_sentence_count": 0,
        }

    budget = max(max_chars, CONTEXT_COMPRESSION_MIN_BLOCK_CHARS)
    query_terms = _normalize_terms(query)
    if not query_terms and not query_type:
        compressed = original[:budget]
        return compressed, {
            "before_chars": len(original),
            "after_chars": len(compressed),
            "sentence_count": len(sentences),
            "selected_sentence_count": len(sentences),
        }

    scored = [
        (
            _score_sentence(
                sentence=sentence,
                sentence_index=index,
                query_terms=query_terms,
                query_type=query_type,
            ),
            index,
            sentence,
        )
        for index, sentence in enumerate(sentences)
    ]
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))

    selected_indexes: list[int] = []
    selected_chars = 0
    for score, index, sentence in ranked:
        if score <= 0 and selected_indexes:
            continue
        if len(selected_indexes) >= CONTEXT_COMPRESSION_MAX_SENTENCES_PER_BLOCK:
            break
        if selected_chars + len(sentence) > budget and selected_indexes:
            continue
        selected_indexes.append(index)
        selected_chars += len(sentence)

    if not selected_indexes:
        selected_indexes = [0]

    selected_sentences = [sentences[index] for index in sorted(selected_indexes)]
    compressed = " ".join(selected_sentences).strip()
    if len(compressed) > budget:
        compressed = compressed[:budget]

    return compressed, {
        "before_chars": len(original),
        "after_chars": len(compressed),
        "sentence_count": len(sentences),
        "selected_sentence_count": len(selected_sentences),
    }


def compress_memory_context(
    memory_hits: list[dict],
    *,
    query: str = "",
    query_type: str = "",
) -> tuple[str, dict]:
    """对 memory 命中做轻量句子级压缩，复用 doc 侧的压缩逻辑。

    多轮对话时 memory_hits 原始内容可能长达数百字（完整的 Q&A 对），
    直接全量拼接会撑满 context 预算，并引入和当前问题无关的噪声句子。

    对每条 hit 独立做句子评分选优，再用总字符上限做第二道截断。
    """

    hits_to_use = memory_hits[:MEMORY_COMPRESSION_MAX_HITS]
    compressed_parts: list[str] = []
    block_stats: list[dict] = []
    total_chars = 0

    for i, hit in enumerate(hits_to_use):
        content = hit.get("content", "").strip()
        if not content:
            continue

        remaining_budget = MEMORY_COMPRESSION_MAX_TOTAL_CHARS - total_chars
        if remaining_budget <= 0:
            break

        block_max = min(MEMORY_COMPRESSION_MAX_BLOCK_CHARS, remaining_budget)
        compressed, stats = _compress_block_content(
            content=content,
            query=query,
            query_type=query_type,
            max_chars=block_max,
        )
        if compressed:
            compressed_parts.append(compressed)
            total_chars += len(compressed)
            block_stats.append({"hit_index": i, **stats})

    compressed_text = "\n".join(compressed_parts)
    before_chars = sum(s["before_chars"] for s in block_stats)
    after_chars = sum(s["after_chars"] for s in block_stats)
    return compressed_text, {
        "enabled": True,
        "hits_used": len(compressed_parts),
        "hits_available": len(memory_hits),
        "before_chars": before_chars,
        "after_chars": after_chars,
        "compression_ratio": (
            round(after_chars / before_chars, 4) if before_chars else 0.0
        ),
        "blocks": block_stats,
    }


def compress_doc_context(doc_hits: list[dict]) -> str:
    """按默认 RAG 配置压缩文档上下文。"""

    return compress_doc_context_with_limit(
        doc_hits,
        RAG_CONFIG.max_doc_context_chars,
    )


def compress_doc_context_with_limit(
    doc_hits: list[dict],
    max_chars: int,
    *,
    query: str = "",
    query_type: str = "",
) -> str:
    """按字符上限截断文档上下文，避免生成阶段输入过长。

    生成阶段通常是最贵的一步，这里先用轻量截断压缩上下文，
    避免把整段长文档原样塞给模型，导致生成耗时和 token 成本过高。
    """

    compressed_context, _stats = compress_doc_context_with_stats(
        doc_hits,
        max_chars,
        query=query,
        query_type=query_type,
    )
    return compressed_context


def compress_doc_context_with_stats(
    doc_hits: list[dict],
    max_chars: int,
    *,
    query: str = "",
    query_type: str = "",
) -> tuple[str, dict]:
    """生成带引用的压缩上下文，并返回压缩统计。"""

    citations = build_citations(doc_hits, RAG_CONFIG.max_doc_context_blocks)
    selected_blocks = []
    citation_index = 0
    block_stats = []

    for doc in doc_hits[: RAG_CONFIG.max_doc_context_blocks]:
        content = doc.get("content", "").strip()
        if not content:
            continue
        citation = citations[citation_index]
        citation_index += 1
        compressed_content, stats = _compress_block_content(
            content=content,
            query=query,
            query_type=query_type,
            max_chars=max_chars,
        )
        block_doc = {**doc, "content": compressed_content}
        selected_blocks.append(format_cited_doc_block(block_doc, citation, max_chars))
        block_stats.append(
            {
                "ref": citation["ref"],
                "doc_id": citation.get("doc_id", ""),
                **stats,
            }
        )

    before_chars = sum(item["before_chars"] for item in block_stats)
    after_chars = sum(item["after_chars"] for item in block_stats)
    ratio = round(after_chars / before_chars, 4) if before_chars else 0.0
    return "\n".join(selected_blocks), {
        "enabled": True,
        "before_chars": before_chars,
        "after_chars": after_chars,
        "compression_ratio": ratio,
        "blocks": block_stats,
    }


def build_rag_context(
    *,
    doc_hits: list[dict],
    memory_hits: list[dict],
    doc_context_chars: int,
    query: str = "",
    query_type: str = "",
) -> RagContext:
    """构建 RAG 生成阶段需要的完整上下文。

    context 是 doc/memory 的完整拼接版本；doc_context 和 memory_context
    分别保留给不同回答分支使用，避免 node.py 里散落字符串拼接。

    memory_hits 经过与 doc_hits 相同级别的句子压缩：每条命中有字符上限，
    整体有总量预算，防止多轮积累的历史噪声撑满 token。
    """

    citations = build_citations(doc_hits, RAG_CONFIG.max_doc_context_blocks)
    doc_context, compression_stats = compress_doc_context_with_stats(
        doc_hits,
        doc_context_chars,
        query=query,
        query_type=query_type,
    )
    memory_context, memory_compression_stats = compress_memory_context(
        memory_hits,
        query=query,
        query_type=query_type,
    )

    context = ""
    if doc_hits:
        context += "知识库资料：\n"
        context += doc_context + "\n"

    if memory_hits:
        context += "\n历史相关记录：\n"
        context += memory_context + "\n"

    return RagContext(
        context=context,
        doc_context=doc_context,
        memory_context=memory_context,
        citations=citations,
        context_compression=compression_stats,
        memory_compression=memory_compression_stats,
    )

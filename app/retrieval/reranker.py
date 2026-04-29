# app/retrieval/reranker.py

from app.llm import LLMCallError, chat
from app.prompts.tooling import build_rerank_prompt
from app.utils.logger import log_warning

RERANK_CONTENT_MAX_CHARS = 800
RERANK_METADATA_FIELDS = (
    ("doc_title", "标题"),
    ("source", "来源"),
    ("section_title", "章节"),
)
RERANK_SCORE_FIELDS = (
    ("score", "综合分"),
    ("semantic_score", "语义分"),
    ("keyword_score_norm", "关键词分"),
)


def _format_score(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value).strip()


def _truncate_content(content: str, max_chars: int = RERANK_CONTENT_MAX_CHARS) -> str:
    content = content.strip()
    if len(content) <= max_chars:
        return content
    return f"{content[:max_chars].rstrip()}..."


def format_rerank_candidate(
    candidate: dict,
    *,
    max_content_chars: int = RERANK_CONTENT_MAX_CHARS,
) -> str:
    """把候选 chunk 转成带 metadata、score 先验的 rerank 输入文本。"""

    lines = []
    for field, label in RERANK_METADATA_FIELDS:
        value = str(candidate.get(field) or "").strip()
        if value:
            lines.append(f"{label}: {value}")

    score_parts = []
    for field, label in RERANK_SCORE_FIELDS:
        if field not in candidate or candidate.get(field) in (None, ""):
            continue
        score_parts.append(f"{label}: {_format_score(candidate.get(field))}")
    if score_parts:
        lines.append("分数先验: " + " / ".join(score_parts))

    retrieval_sources = candidate.get("retrieval_sources") or []
    if retrieval_sources:
        lines.append(
            "召回来源: " + ", ".join(str(source) for source in retrieval_sources)
        )

    content = _truncate_content(
        str(candidate.get("content") or ""),
        max_chars=max_content_chars,
    )
    if content:
        lines.append(f"正文: {content}")

    return "\n".join(lines)


def rerank(query: str, candidates: list[dict], top_k: int = 2):
    """对候选结果做轻量 LLM 重排。

    这里的定位是"可选增强"而不是"硬依赖"：
    - 成功时提升排序质量
    - 失败时直接退回原始候选，不允许因为 rerank 挂掉把主链路打崩
    """
    if not candidates:
        return []

    texts = [format_rerank_candidate(c) for c in candidates]

    prompt = build_rerank_prompt(query, texts)

    try:
        res = chat([{"role": "user", "content": prompt}])
    except LLMCallError as exc:
        log_warning(
            "rerank",
            "LLM rerank failed; fallback to original candidates",
            {
                "code": exc.code,
                "profile": exc.profile,
                "provider": exc.provider,
                "model": exc.model,
                "candidate_count": len(candidates),
            },
        )
        return candidates[:top_k]

    try:
        import json

        idxs = json.loads(res)
        reranked = [candidates[i] for i in idxs if i < len(candidates)]
        return reranked[:top_k]
    except (json.JSONDecodeError, TypeError, ValueError, IndexError) as exc:
        log_warning(
            "rerank",
            "invalid rerank response; fallback to original candidates",
            {
                "error": f"{exc.__class__.__name__}: {exc}",
                "response_preview": res[:160],
                "candidate_count": len(candidates),
            },
        )
        return candidates[:top_k]

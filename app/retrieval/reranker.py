# app/retrieval/reranker.py

from app.llm import LLMCallError, chat
from app.prompts.tooling import build_rerank_prompt
from app.utils.logger import log_warning


def rerank(query: str, candidates: list[dict], top_k: int = 2):
    """对候选结果做轻量 LLM 重排。

    这里的定位是“可选增强”而不是“硬依赖”：
    - 成功时提升排序质量
    - 失败时直接退回原始候选，不允许因为 rerank 挂掉把主链路打崩
    """
    if not candidates:
        return []

    texts = [c["content"] for c in candidates]

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

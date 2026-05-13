"""Phase 3 PR-3：RAG 检索 + 回答的纯函数封装。

职责边界：
- **不触 AgentState**：只接受已就绪的 query / strategy / on_delta，返回结构化结果。
- **只走文档分支**：multi_hop 场景下不再兜底 memory（历史记忆与多跳证据语义
  不同，混进来反而噪声），也不走 rewrite（rewrite 已在 rag_agent_node 或
  multi_hop_node 入口统一完成）。
- **不写 State / 不写 debug_info / 不触 agent_outputs**：由调用方（multi_hop_node）
  自己决定如何落盘，避免"fallback 嵌套 rag_agent_node"那种双写风险。

两个用法：
- `run_doc_retrieval_and_answer(query, query_type=..., strategy=...)`：核心内
  部调用，multi_hop_node 每跳与最终 answer 都复用这里的回答逻辑。
- `run_single_hop_retrieval_answer(base_query, auth=...)`：decompose 失败时的
  兜底入口——把 multi_hop 降级到"单跳 retrieve + 回答"，输出结构与正常多跳
  答案形态兼容，方便合成 pseudo-step。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from app.agents.rag.answer import generate_answer_for_context
from app.agents.rag.constants import ANSWER_STRATEGY_MULTI_HOP
from app.agents.rag.context import build_rag_context
from app.agents.rag.doc_pipeline import retrieve_docs_for_rag
from app.agents.rag.query_classifier import classify_rag_query
from app.agents.rag.strategy import (
    adapt_strategy_max_tokens,
    build_doc_answer_strategy,
)
from app.agents.rag.types import QueryClassification
from app.config import RAG_CONFIG
from app.utils.logger import now_ms


@dataclass(frozen=True)
class SingleHopAnswerResult:
    """`run_single_hop_retrieval_answer` 的结构化返回。

    multi_hop_node 的 fallback 分支按此结构写 pseudo-step / citations / answer，
    保持和正常多跳答案同样的字段形状，避免上游判空要写两套。
    """

    answer: str
    citations: list[dict]
    doc_hits: list[dict]
    merged_doc_hits: list[dict]
    retrieval_debug: dict
    answer_strategy: dict
    errors: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


def _build_multi_hop_answer_strategy() -> dict:
    """Multi-hop 回答策略：明确分 subquery 叙述 + 统一 citations。

    - context_chars 取单跳上限，避免把多跳的多个 sq 证据压成过短摘要。
    - max_tokens 走单跳上限，再由 `adapt_strategy_max_tokens` 按 context 长度
      动态收紧，防止证据少时生成过长的"凑字数"答案。
    - answer_style 要求 LLM 分节点叙述，让下游可按 citation 回溯到具体 sq。
    """

    return {
        "name": ANSWER_STRATEGY_MULTI_HOP,
        "answer_style": (
            "先按每个子问题分段给出基于资料的事实陈述，"
            "再给综合结论；只使用资料里明确出现的内容，"
            "保留 [N] 引用编号，不要补充常识。"
        ),
        "context_chars": RAG_CONFIG.max_doc_context_chars,
        "max_tokens": RAG_CONFIG.max_doc_answer_tokens,
    }


def build_multi_hop_answer_strategy() -> dict:
    """对外：multi_hop_node 在合成最终 answer 时用此策略。

    单独暴露一个函数是为了让 supervisor / eval 可以显式断言 "multi_hop 答案
    走的就是 ANSWER_STRATEGY_MULTI_HOP"，而不是被 query_type 策略误覆盖。
    """

    return _build_multi_hop_answer_strategy()


def answer_with_doc_hits(
    *,
    question: str,
    doc_hits: list[dict],
    query_type: str = "",
    strategy: dict | None = None,
    on_delta: Callable | None = None,
) -> tuple[str, list[dict], dict, dict, list[str], float]:
    """基于已检索好的 doc_hits 生成答案。

    返回 `(answer, citations, answer_strategy, context_compression, errors, timing_ms)`。
    不涉及任何 AgentState 写入；调用方自行合成 debug / step_results。
    """

    started = now_ms()
    errors: list[str] = []

    if strategy is None:
        strategy = build_doc_answer_strategy(
            question,
            classification=(
                QueryClassification(
                    query_type=query_type,
                    confidence=1.0,
                    reason="answer_flow_default",
                )
                if query_type
                else None
            ),
        )

    rag_context = build_rag_context(
        doc_hits=doc_hits,
        memory_hits=[],
        doc_context_chars=strategy["context_chars"],
        query=question,
        query_type=query_type,
    )
    strategy = adapt_strategy_max_tokens(
        strategy,
        actual_context_chars=len(rag_context.doc_context),
    )

    has_strong_knowledge = bool(doc_hits)
    answer_result = generate_answer_for_context(
        question=question,
        rag_context=rag_context,
        doc_answer_strategy=strategy,
        has_strong_knowledge=has_strong_knowledge,
        has_memory=False,
        on_delta=on_delta,
    )
    errors.extend(answer_result.errors)
    return (
        answer_result.answer,
        rag_context.citations,
        strategy,
        rag_context.context_compression,
        errors,
        round(now_ms() - started, 2),
    )


def run_single_hop_retrieval_answer(
    *,
    base_query: str,
    on_delta: Callable | None = None,
) -> SingleHopAnswerResult:
    """Multi-hop decompose 失败时的单跳兜底。

    - 输入 `base_query` 已是 multi_hop_node 选好的 rewrite 或最新 user message，
      不再二次 rewrite。
    - 只走文档检索，不走 memory。memory 的历史语义在多跳降级场景没价值，
      反而容易稀释证据。
    - 回答策略按 query_type 正常选（和单跳 RAG 保持一致体验），不强制
      ANSWER_STRATEGY_MULTI_HOP——因为实际上只剩一个 subquery，没有多跳叙述需要。
    """

    started = now_ms()
    errors: list[str] = []
    sub_timings_ms: dict[str, float] = {}

    # 单跳 fallback：轻量分类，不触发 LLM 二裁（delay 敏感）
    query_classification = classify_rag_query(
        original_query=base_query,
        rewritten_query=base_query,
        has_context=False,
        llm_fallback=False,
    )

    doc_result = retrieve_docs_for_rag(
        base_query,
        query_type=query_classification.query_type,
        confidence=query_classification.confidence,
    )
    errors.extend(doc_result.errors)
    sub_timings_ms.update(doc_result.timings_ms)

    strategy = build_doc_answer_strategy(
        base_query,
        classification=query_classification,
    )
    answer, citations, strategy, _compression, ans_errors, ans_ms = (
        answer_with_doc_hits(
            question=base_query,
            doc_hits=doc_result.merged_doc_hits,
            query_type=query_classification.query_type,
            strategy=strategy,
            on_delta=on_delta,
        )
    )
    errors.extend(ans_errors)
    sub_timings_ms["answerGeneration"] = ans_ms
    sub_timings_ms["singleHopTotal"] = round(now_ms() - started, 2)

    return SingleHopAnswerResult(
        answer=answer,
        citations=citations,
        doc_hits=doc_result.merged_doc_hits,
        merged_doc_hits=doc_result.merged_doc_hits,
        retrieval_debug=doc_result.retrieval_debug,
        answer_strategy=strategy,
        errors=errors,
        timings_ms=sub_timings_ms,
    )


__all__ = [
    "SingleHopAnswerResult",
    "answer_with_doc_hits",
    "build_multi_hop_answer_strategy",
    "run_single_hop_retrieval_answer",
]

from app.agents.rag.constants import (
    ANSWER_STRATEGY_COMPARISON,
    ANSWER_STRATEGY_FALLBACK,
    ANSWER_STRATEGY_FOLLOWUP,
    QUERY_TYPE_COMPARISON,
    QUERY_TYPE_DEFINITION,
    QUERY_TYPE_FACTUAL,
    QUERY_TYPE_FALLBACK,
    QUERY_TYPE_FOLLOWUP,
)
from app.agents.rag.query_classifier import classify_rag_query
from app.agents.rag.strategy import build_doc_answer_strategy
from app.constants.policies import ANSWER_STRATEGY_DEFINITION_SHORT


def _classify(
    original: str,
    rewritten: str | None = None,
    *,
    has_context: bool = False,
):
    return classify_rag_query(
        original_query=original,
        rewritten_query=rewritten or original,
        has_context=has_context,
    )


def test_classifies_definition_query() -> None:
    result = _classify("WAI-ARIA技术是什么")

    assert result.query_type == QUERY_TYPE_DEFINITION
    assert result.confidence > 0.8


def test_classifies_comparison_query() -> None:
    result = _classify("虚拟列表和分页有什么区别")

    assert result.query_type == QUERY_TYPE_COMPARISON


def test_classifies_followup_from_original_query() -> None:
    result = _classify(
        "那上海呢",
        rewritten="上海气候怎么样？",
        has_context=True,
    )

    assert result.query_type == QUERY_TYPE_FOLLOWUP


def test_classifies_pronoun_query_as_followup_with_context() -> None:
    result = _classify("它怎么配置", has_context=True)

    assert result.query_type == QUERY_TYPE_FOLLOWUP


def test_classifies_vague_instruction_as_followup_with_context() -> None:
    result = _classify("介绍一下", has_context=True)

    assert result.query_type == QUERY_TYPE_FOLLOWUP


def test_classifies_vague_query_as_fallback_without_context() -> None:

    result = _classify("介绍一下", has_context=False)

    assert result.query_type == QUERY_TYPE_FALLBACK


def test_classifies_default_factual_query() -> None:
    result = _classify("北京气候怎么样")

    assert result.query_type == QUERY_TYPE_FACTUAL


def test_strategy_follows_query_classification() -> None:
    definition = build_doc_answer_strategy(
        "WAI-ARIA技术是什么",
        classification=_classify("WAI-ARIA技术是什么"),
    )
    comparison = build_doc_answer_strategy(
        "虚拟列表和分页有什么区别",
        classification=_classify("虚拟列表和分页有什么区别"),
    )
    followup = build_doc_answer_strategy(
        "上海气候怎么样？",
        classification=_classify("那上海呢", "上海气候怎么样？", has_context=True),
    )
    fallback = build_doc_answer_strategy(
        "介绍一下",
        classification=_classify("介绍一下"),
    )

    assert definition["name"] == ANSWER_STRATEGY_DEFINITION_SHORT
    assert comparison["name"] == ANSWER_STRATEGY_COMPARISON
    assert followup["name"] == ANSWER_STRATEGY_FOLLOWUP
    assert fallback["name"] == ANSWER_STRATEGY_FALLBACK

from app.agents.rag.constants import (
    ANSWER_STRATEGY_COMPARISON,
    ANSWER_STRATEGY_FALLBACK,
    ANSWER_STRATEGY_FOLLOWUP,
    CLASSIFIER_LLM_CONFIDENCE_THRESHOLD,
    QUERY_TYPE_COMPARISON,
    QUERY_TYPE_DEFINITION,
    QUERY_TYPE_FACTUAL,
    QUERY_TYPE_FALLBACK,
    QUERY_TYPE_FOLLOWUP,
)
from app.agents.rag.query_classifier import (
    _parse_llm_classification,
    _should_llm_classify,
    classify_rag_query,
)
from app.agents.rag.strategy import build_doc_answer_strategy
from app.agents.rag.types import QueryClassification
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


def test_classifies_howto_query_as_definition() -> None:
    # "怎么用/如何配置" 应分为 definition，驱动定义型策略而非 factual 兜底
    assert _classify("Redis 怎么用").query_type == QUERY_TYPE_DEFINITION
    assert _classify("JWT 如何配置").query_type == QUERY_TYPE_DEFINITION
    assert _classify("如何设置过期时间").query_type == QUERY_TYPE_DEFINITION
    assert _classify("这个有什么作用").query_type == QUERY_TYPE_DEFINITION


def test_classifies_howto_with_pronoun_as_followup_with_context() -> None:
    # "它怎么用" 带上下文时 FOLLOWUP 优先于 DEFINITION（"它"是追问信号）
    assert _classify("它怎么用", has_context=True).query_type == QUERY_TYPE_FOLLOWUP


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


# ---------------------------------------------------------------------------
# LLM fallback 单测
# ---------------------------------------------------------------------------


def test_should_llm_classify_true_for_low_confidence() -> None:
    low = QueryClassification(
        query_type=QUERY_TYPE_FACTUAL, confidence=0.6, reason="default_factual_query"
    )
    assert _should_llm_classify(low) is True


def test_should_llm_classify_false_for_high_confidence() -> None:
    high = QueryClassification(
        query_type=QUERY_TYPE_DEFINITION,
        confidence=0.9,
        reason="query_contains_definition_signal",
    )
    assert _should_llm_classify(high) is False


def test_should_llm_classify_threshold_boundary() -> None:
    # 恰好等于阈值不触发
    at_threshold = QueryClassification(
        query_type=QUERY_TYPE_FACTUAL,
        confidence=CLASSIFIER_LLM_CONFIDENCE_THRESHOLD,
        reason="default_factual_query",
    )
    assert _should_llm_classify(at_threshold) is False


def test_parse_llm_classification_valid_json() -> None:
    result = _parse_llm_classification('{"type": "definition", "confidence": 0.92}')
    assert result is not None
    assert result.query_type == QUERY_TYPE_DEFINITION
    assert result.confidence == 0.92
    assert result.reason == "llm_classifier"


def test_parse_llm_classification_invalid_type_returns_none() -> None:
    result = _parse_llm_classification('{"type": "unknown_type", "confidence": 0.9}')
    assert result is None


def test_parse_llm_classification_malformed_json_returns_none() -> None:
    assert _parse_llm_classification("not json at all") is None
    assert _parse_llm_classification("") is None


def test_parse_llm_classification_clips_confidence() -> None:
    # LLM 输出越界时应裁剪到合法范围
    result = _parse_llm_classification('{"type": "factual", "confidence": 1.5}')
    assert result is not None
    assert result.confidence == 1.0

    result2 = _parse_llm_classification('{"type": "factual", "confidence": -0.1}')
    assert result2 is not None
    assert result2.confidence == 0.0


def test_classify_rag_query_llm_fallback_triggers_on_low_confidence(
    monkeypatch,
) -> None:
    # FACTUAL（confidence=0.6）应触发 LLM 二裁，LLM 返回 definition
    def fake_classify_with_llm(query: str):
        return QueryClassification(
            query_type=QUERY_TYPE_DEFINITION,
            confidence=0.88,
            reason="llm_classifier",
        )

    import app.agents.rag.query_classifier as qc_mod

    monkeypatch.setattr(qc_mod, "classify_with_llm", fake_classify_with_llm)

    result = classify_rag_query(
        original_query="北京气候怎么样",
        rewritten_query="北京气候怎么样？",
        has_context=False,
        llm_fallback=True,
    )
    assert result.query_type == QUERY_TYPE_DEFINITION
    assert result.reason == "llm_classifier"


def test_classify_rag_query_llm_fallback_skipped_when_disabled(monkeypatch) -> None:
    # llm_fallback=False 时不调 LLM，直接用规则结果
    called = {"count": 0}

    def fake_classify_with_llm(query: str):
        called["count"] += 1
        return None

    import app.agents.rag.query_classifier as qc_mod

    monkeypatch.setattr(qc_mod, "classify_with_llm", fake_classify_with_llm)

    result = classify_rag_query(
        original_query="北京气候怎么样",
        rewritten_query="北京气候怎么样？",
        has_context=False,
        llm_fallback=False,
    )
    assert called["count"] == 0
    assert result.query_type == QUERY_TYPE_FACTUAL


def test_classify_rag_query_llm_failure_falls_back_to_rule(monkeypatch) -> None:
    # LLM 返回 None（失败）时，规则结果兜底
    import app.agents.rag.query_classifier as qc_mod

    monkeypatch.setattr(qc_mod, "classify_with_llm", lambda q: None)

    result = classify_rag_query(
        original_query="北京气候怎么样",
        rewritten_query="北京气候怎么样？",
        has_context=False,
        llm_fallback=True,
    )
    assert result.query_type == QUERY_TYPE_FACTUAL
    assert result.reason == "default_factual_query"


def test_classify_rag_query_high_confidence_skips_llm(monkeypatch) -> None:
    # 规则置信度高（DEFINITION=0.9）时不触发 LLM
    called = {"count": 0}

    def fake_classify_with_llm(query: str):
        called["count"] += 1
        return None

    import app.agents.rag.query_classifier as qc_mod

    monkeypatch.setattr(qc_mod, "classify_with_llm", fake_classify_with_llm)

    result = classify_rag_query(
        original_query="Redis 是什么",
        rewritten_query="Redis 是什么？",
        has_context=False,
        llm_fallback=True,
    )
    assert called["count"] == 0
    assert result.query_type == QUERY_TYPE_DEFINITION

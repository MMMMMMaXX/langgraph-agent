"""Unit tests for `app/agents/rag/multi_hop/gate.py`。

重点回归 `docs/phase3-multi-hop-rag.md` §2.3 / §10 约定的负向门控必过样本。
gate 的契约：negative 优先 → positive trigger → 否则 False。
"""

from __future__ import annotations

import pytest

from app.agents.rag.multi_hop.gate import (
    matches_negative_gate,
    matches_positive_trigger,
    preprocess_query_for_gate,
    should_enter_multi_hop,
)


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def test_preprocess_strips_whitespace_and_trailing_punct() -> None:
    assert preprocess_query_for_gate("  什么是 JWT？  ") == "什么是 JWT"
    assert preprocess_query_for_gate("WAI-ARIA 和虚拟列表有什么区别？？") == (
        "WAI-ARIA 和虚拟列表有什么区别"
    )
    assert preprocess_query_for_gate("X   vs   Y") == "X vs Y"


def test_preprocess_empty_returns_empty() -> None:
    assert preprocess_query_for_gate("") == ""
    assert preprocess_query_for_gate("   ") == ""
    assert preprocess_query_for_gate(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# negative gate：§2.3 列举的 7 个必过样本
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "WAI-ARIA 和虚拟列表有什么区别？",
        "WAI-ARIA vs 虚拟列表",
        "React 与 Vue 的区别是什么",
        "Redis 对比 Memcached",
        "Kafka 和 Pulsar 哪个更稳定？",
        "什么是 JWT？",
        "OAuth 是什么",
    ],
)
def test_negative_gate_blocks_simple_comparison_samples(query: str) -> None:
    assert matches_negative_gate(query), f"negative gate 漏拦：{query}"
    assert should_enter_multi_hop(query) is False


def test_negative_gate_blocks_empty_query() -> None:
    assert matches_negative_gate("") is True
    assert should_enter_multi_hop("") is False


# ---------------------------------------------------------------------------
# positive trigger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "基于 A 项目接口文档和 B 项目部署逻辑生成一个上线方案",
        "结合用户中心的鉴权流程以及订单系统的退款流程，写一份对账指南",
        "先查一下昨天的监控错误码，再根据它推导可能的根因",
        "根据 A 系统的 QPS 指标分析 B 系统的慢查询来源",
        "跨项目排查登录失败问题",
    ],
)
def test_positive_trigger_matches_cross_doc_chain_samples(query: str) -> None:
    # 这些样本都不应被 negative gate 挡住
    assert matches_negative_gate(query) is False
    assert matches_positive_trigger(query) is True
    assert should_enter_multi_hop(query) is True


def test_negative_gate_overrides_positive_trigger() -> None:
    """即使命中正向触发，negative gate 也应优先生效。

    虽然当前样本较难同时命中，但为了锁死优先级契约在此构造人工样本。
    """

    # "根据 A 的指标分析 B" 触发 trigger；但如果改写成"X 对比 Y"就该被挡。
    assert should_enter_multi_hop("A 对比 B") is False


def test_non_matching_query_returns_false() -> None:
    # 普通单跳 query：既不触发 negative 也不触发 positive，返回 False
    # 由 supervisor 决定走 LLM 兜底分类。
    assert should_enter_multi_hop("帮我查下北京今天的天气") is False
    assert matches_negative_gate("帮我查下北京今天的天气") is False
    assert matches_positive_trigger("帮我查下北京今天的天气") is False

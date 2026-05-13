"""Unit tests for `app/agents/rag/multi_hop/decompose.py`。

覆盖：
- schema pass 正常产出 Subquery tuple；
- schema fail / 非法 intent / 越界 / 自引用 / 空子查询 → 降级单跳；
- 同义单子查询 → 降级单跳（error_code=SYNONYM_SUBQUERY）；
- LLM 调用失败（LLMCallError）→ 降级单跳；
- depends_on 引用未出现的 id → 降级单跳。
"""

from __future__ import annotations

import json

import pytest

from app.agents.rag.multi_hop import decompose as decompose_mod
from app.agents.rag.multi_hop.decompose import decompose_query
from app.constants.multi_hop import (
    DEGRADE_REASON_DECOMPOSE_FAILED,
    DEGRADE_REASON_SYNONYM_SUBQUERY,
    MAX_SUBQUERIES,
)
from app.llm.retry import LLMCallError

# ---------------------------------------------------------------------------
# 成功路径
# ---------------------------------------------------------------------------


def test_decompose_happy_path(llm_stub) -> None:
    payload = {
        "subqueries": [
            {
                "id": "sq1",
                "intent": "entity_lookup",
                "query": "A 项目接口文档的鉴权流程",
                "depends_on": [],
            },
            {
                "id": "sq2",
                "intent": "procedure",
                "query": "B 项目部署逻辑的回滚步骤",
                "depends_on": ["sq1"],
            },
        ]
    }
    llm_stub.set_response(json.dumps(payload))

    result = decompose_query(
        rewritten_query="基于 A 项目和 B 项目写一份上线方案",
        role="user",
    )

    assert result.degraded_to_single_hop is False
    assert result.error_code == ""
    assert [sq.id for sq in result.subqueries] == ["sq1", "sq2"]
    assert result.subqueries[1].depends_on == ("sq1",)
    assert result.subqueries[0].intent == "entity_lookup"


def test_decompose_intent_normalized_to_lowercase(llm_stub) -> None:
    payload = {
        "subqueries": [
            {"id": "sq1", "intent": "Entity_Lookup", "query": "X", "depends_on": []}
        ]
    }
    llm_stub.set_response(json.dumps(payload))
    # 注意这里用不同的 rewritten，否则会被 synonym 检测判退化
    result = decompose_query(
        rewritten_query="完整的跨项目上线方案",
        role="user",
    )
    assert result.degraded_to_single_hop is False
    assert result.subqueries[0].intent == "entity_lookup"


# ---------------------------------------------------------------------------
# 降级路径：schema 失败
# ---------------------------------------------------------------------------


def test_decompose_non_json_response_degrades(llm_stub) -> None:
    llm_stub.set_response("抱歉，我做不到。")
    result = decompose_query(rewritten_query="基于 A 和 B 生成方案", role="user")

    assert result.degraded_to_single_hop is True
    assert result.error_code == DEGRADE_REASON_DECOMPOSE_FAILED
    # fallback 保证有且只有 1 个兜底 subquery，等同原 query
    assert len(result.subqueries) == 1
    assert result.subqueries[0].query == "基于 A 和 B 生成方案"


def test_decompose_invalid_intent_degrades(llm_stub) -> None:
    llm_stub.set_response(
        json.dumps(
            {
                "subqueries": [
                    {
                        "id": "sq1",
                        "intent": "illegal_intent",
                        "query": "X",
                        "depends_on": [],
                    }
                ]
            }
        )
    )
    result = decompose_query(rewritten_query="复杂跨系统方案", role="user")
    assert result.degraded_to_single_hop is True
    assert result.error_code == DEGRADE_REASON_DECOMPOSE_FAILED


def test_decompose_exceeds_max_subqueries_degrades(llm_stub) -> None:
    payload = {
        "subqueries": [
            {
                "id": f"sq{i}",
                "intent": "entity_lookup",
                "query": f"Q{i}",
                "depends_on": [],
            }
            for i in range(1, MAX_SUBQUERIES + 2)
        ]
    }
    llm_stub.set_response(json.dumps(payload))
    result = decompose_query(rewritten_query="跨项目方案", role="user")
    assert result.degraded_to_single_hop is True
    assert result.error_code == DEGRADE_REASON_DECOMPOSE_FAILED


def test_decompose_duplicate_id_degrades(llm_stub) -> None:
    llm_stub.set_response(
        json.dumps(
            {
                "subqueries": [
                    {
                        "id": "sq1",
                        "intent": "entity_lookup",
                        "query": "X",
                        "depends_on": [],
                    },
                    {
                        "id": "sq1",
                        "intent": "procedure",
                        "query": "Y",
                        "depends_on": [],
                    },
                ]
            }
        )
    )
    result = decompose_query(rewritten_query="跨项目方案", role="user")
    assert result.degraded_to_single_hop is True


def test_decompose_depends_on_unknown_id_degrades(llm_stub) -> None:
    llm_stub.set_response(
        json.dumps(
            {
                "subqueries": [
                    {
                        "id": "sq1",
                        "intent": "entity_lookup",
                        "query": "X",
                        "depends_on": ["sq99"],
                    }
                ]
            }
        )
    )
    result = decompose_query(rewritten_query="跨项目方案", role="user")
    assert result.degraded_to_single_hop is True


def test_decompose_self_dependency_degrades(llm_stub) -> None:
    llm_stub.set_response(
        json.dumps(
            {
                "subqueries": [
                    {
                        "id": "sq1",
                        "intent": "entity_lookup",
                        "query": "X",
                        "depends_on": ["sq1"],
                    }
                ]
            }
        )
    )
    result = decompose_query(rewritten_query="跨项目方案", role="user")
    assert result.degraded_to_single_hop is True


def test_decompose_empty_subqueries_degrades(llm_stub) -> None:
    llm_stub.set_response(json.dumps({"subqueries": []}))
    result = decompose_query(rewritten_query="跨项目方案", role="user")
    assert result.degraded_to_single_hop is True


def test_decompose_sq2_only_degrades(llm_stub) -> None:
    """单独出现 sq2（未从 sq1 开始）应当降级。"""

    llm_stub.set_response(
        json.dumps(
            {
                "subqueries": [
                    {
                        "id": "sq2",
                        "intent": "entity_lookup",
                        "query": "X",
                        "depends_on": [],
                    }
                ]
            }
        )
    )
    result = decompose_query(rewritten_query="跨项目方案", role="user")
    assert result.degraded_to_single_hop is True
    assert result.error_code == DEGRADE_REASON_DECOMPOSE_FAILED


def test_decompose_sq1_sq3_skip_degrades(llm_stub) -> None:
    """sq1 -> sq3 跳号（缺 sq2）应当降级。"""

    llm_stub.set_response(
        json.dumps(
            {
                "subqueries": [
                    {
                        "id": "sq1",
                        "intent": "entity_lookup",
                        "query": "X",
                        "depends_on": [],
                    },
                    {
                        "id": "sq3",
                        "intent": "procedure",
                        "query": "Y",
                        "depends_on": ["sq1"],
                    },
                ]
            }
        )
    )
    result = decompose_query(rewritten_query="跨项目方案", role="user")
    assert result.degraded_to_single_hop is True
    assert result.error_code == DEGRADE_REASON_DECOMPOSE_FAILED


# ---------------------------------------------------------------------------
# 降级路径：同义退化（单 subquery 与 rewritten 等价）
# ---------------------------------------------------------------------------


def test_decompose_single_synonym_subquery_degrades(llm_stub) -> None:
    llm_stub.set_response(
        json.dumps(
            {
                "subqueries": [
                    {
                        "id": "sq1",
                        "intent": "entity_lookup",
                        "query": "跨项目方案",
                        "depends_on": [],
                    }
                ]
            }
        )
    )
    result = decompose_query(rewritten_query="跨项目方案", role="user")
    assert result.degraded_to_single_hop is True
    assert result.error_code == DEGRADE_REASON_SYNONYM_SUBQUERY


def test_decompose_single_synonym_with_punct_still_degrades(llm_stub) -> None:
    """同义判定要无视空白和常见标点。"""

    llm_stub.set_response(
        json.dumps(
            {
                "subqueries": [
                    {
                        "id": "sq1",
                        "intent": "entity_lookup",
                        "query": " 跨项目方案 ！",
                        "depends_on": [],
                    }
                ]
            }
        )
    )
    result = decompose_query(rewritten_query="跨项目方案", role="user")
    assert result.degraded_to_single_hop is True
    assert result.error_code == DEGRADE_REASON_SYNONYM_SUBQUERY


# ---------------------------------------------------------------------------
# 降级路径：LLM 调用失败
# ---------------------------------------------------------------------------


def test_decompose_llm_failure_degrades(llm_stub, monkeypatch) -> None:
    def _raise(*_args, **_kwargs):
        raise LLMCallError(
            code="timeout",
            message="request timed out",
            profile="routing",
            provider="deepseek",
            model="deepseek-chat",
        )

    monkeypatch.setattr(decompose_mod, "_call_decompose_llm", _raise)
    result = decompose_query(rewritten_query="跨项目方案", role="user")
    assert result.degraded_to_single_hop is True
    assert result.error_code == DEGRADE_REASON_DECOMPOSE_FAILED


def test_decompose_empty_rewritten_query_degrades() -> None:
    result = decompose_query(rewritten_query="   ", role="user")
    assert result.degraded_to_single_hop is True
    assert result.error_code == DEGRADE_REASON_DECOMPOSE_FAILED


# ---------------------------------------------------------------------------
# 约定：降级时 fallback subquery 等价于 rewritten
# ---------------------------------------------------------------------------


def test_decompose_fallback_preserves_rewritten_query(llm_stub) -> None:
    llm_stub.set_response("not json")
    result = decompose_query(rewritten_query="根据 A 分析 B", role="user")
    assert result.degraded_to_single_hop is True
    assert len(result.subqueries) == 1
    assert result.subqueries[0].id == "sq1"
    assert result.subqueries[0].query == "根据 A 分析 B"

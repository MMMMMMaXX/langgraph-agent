"""Multi-hop 子查询分解器（Phase 3 PR-1）。

职责边界：
- 输入：rewritten_query + auth.role；
- 输出：`DecomposeResult`（tuple[Subquery, ...] + 降级标记 + 错误码）。
- 失败策略：**降级优先**，不 fail-closed。LLM 调用失败 / JSON 解析失败 / schema
  校验失败 / 生成的子查询与原 query 同义（退化无增益）→ 返回 `degraded_to_single_hop=True`，
  由 multi_hop_node 走 fallback_to_single_hop 分支。这样 multi-hop 不会比单跳更差。

本模块是**纯函数**：不依赖 AgentState / LangGraph。由 multi_hop_node（PR-3）调用。
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from app.agents.rag.multi_hop.types import DecomposeResult, Subquery
from app.constants.model_profiles import PROFILE_ROUTING
from app.constants.multi_hop import (
    DEGRADE_REASON_DECOMPOSE_FAILED,
    DEGRADE_REASON_SYNONYM_SUBQUERY,
    MAX_SUBQUERIES,
    SUBQUERY_INTENT_ENTITY_LOOKUP,
    VALID_SUBQUERY_INTENTS,
)
from app.llm.retry import LLMCallError
from app.prompts.rag import (
    build_decompose_system_prompt,
    build_decompose_user_prompt,
)
from app.utils.logger import log_warning

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


# sq id 严格格式：sq + 十进制正整数（无前导 0）。合法示例：sq1 / sq2 / sq10。
# 非法示例：sq / sq0 / sq01 / squirrel / sq1a。
_SUBQUERY_ID_PATTERN = re.compile(r"^sq[1-9]\d*$")


class _SubqueryModel(BaseModel):
    """单条 subquery 的 schema。"""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    id: str
    intent: str
    query: str
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_must_match_sq_pattern(cls, value: str) -> str:
        if not _SUBQUERY_ID_PATTERN.match(value or ""):
            raise ValueError(f"subquery id must match '^sq[1-9]\\d*$': {value!r}")
        return value

    @field_validator("intent")
    @classmethod
    def _intent_must_be_known(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in VALID_SUBQUERY_INTENTS:
            raise ValueError(
                f"subquery intent must be one of {VALID_SUBQUERY_INTENTS}: "
                f"{value!r}"
            )
        return normalized

    @field_validator("query")
    @classmethod
    def _query_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("subquery query must be non-empty")
        return value.strip()


class _DecomposeModel(BaseModel):
    """Decomposer 顶层 schema。"""

    model_config = ConfigDict(extra="ignore")

    subqueries: list[_SubqueryModel] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_ids_and_deps(self) -> _DecomposeModel:
        if len(self.subqueries) > MAX_SUBQUERIES:
            raise ValueError(
                f"too many subqueries: {len(self.subqueries)} > {MAX_SUBQUERIES}"
            )
        # id 必须恰好等于 ["sq1", "sq2", ...]：拒 sq2-only、sq1/sq3 跳号等情况，
        # 保证 depends_on 引用的前序 id 与输出顺序严格对齐。
        expected_ids = [f"sq{i}" for i in range(1, len(self.subqueries) + 1)]
        actual_ids = [sq.id for sq in self.subqueries]
        if actual_ids != expected_ids:
            raise ValueError(
                f"subquery ids must be sequential sq1..sqN, "
                f"got {actual_ids} expected {expected_ids}"
            )
        seen: set[str] = set()
        for sq in self.subqueries:
            seen.add(sq.id)
            for dep in sq.depends_on:
                if dep == sq.id:
                    raise ValueError(f"subquery {sq.id} depends on itself")
                if dep not in seen:
                    # depends_on 只能指前序已出现的 id（天然无环）
                    raise ValueError(
                        f"subquery {sq.id} depends on unknown or later id: {dep}"
                    )
        return self


class DecomposeParseError(Exception):
    """内部：schema/JSON 解析失败。调用方转 DecomposeResult.degraded。"""


def _parse_decompose_payload(raw: str) -> tuple[Subquery, ...]:
    """把 LLM 原文解析成 `tuple[Subquery, ...]`。失败抛 DecomposeParseError。"""

    try:
        payload: Any = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise DecomposeParseError(f"json_decode_failed: {exc}") from exc

    try:
        model = _DecomposeModel.model_validate(payload)
    except ValidationError as exc:
        raise DecomposeParseError(f"schema_invalid: {exc.errors()[:2]}") from exc

    return tuple(
        Subquery(
            id=sq.id,
            intent=sq.intent,
            query=sq.query,
            depends_on=tuple(sq.depends_on),
        )
        for sq in model.subqueries
    )


# ---------------------------------------------------------------------------
# 降级判定
# ---------------------------------------------------------------------------


def _normalize_for_similarity(text: str) -> str:
    """做最轻量归一：去空白、去常见标点。用于 synonym 判定。"""

    stripped = (text or "").strip().lower()
    # 去掉空白和常见中英文标点
    for ch in (" ", "\t", "\n", "?", "？", "。", ".", "!", "！", ",", "，", "、"):
        stripped = stripped.replace(ch, "")
    return stripped


def _is_synonym_degradation(
    subqueries: tuple[Subquery, ...], rewritten_query: str
) -> bool:
    """判定 Decomposer 是否产出了"与原问题同义"的单一子查询。

    规则：只在 `len==1` 的情况下触发；多 subquery 即使个别同义也不算退化——
    它可能是把原问题作为总览子查询 + 新增若干细分。
    """

    if len(subqueries) != 1:
        return False
    return _normalize_for_similarity(subqueries[0].query) == _normalize_for_similarity(
        rewritten_query
    )


def _fallback_single_hop(
    rewritten_query: str, reason: str, error_code: str = ""
) -> DecomposeResult:
    """构造 fallback 结果：只保留 1 个兜底 subquery，等价于单跳。"""

    fallback_sq = Subquery(
        id="sq1",
        intent=SUBQUERY_INTENT_ENTITY_LOOKUP,
        query=rewritten_query.strip(),
        depends_on=(),
    )
    return DecomposeResult(
        subqueries=(fallback_sq,),
        degraded_to_single_hop=True,
        reason=reason,
        error_code=error_code,
    )


# ---------------------------------------------------------------------------
# LLM 调用 + 主入口
# ---------------------------------------------------------------------------


def _call_decompose_llm(system_prompt: str, user_prompt: str) -> str:
    """走 `_create_chat_completion` 取原文。

    和 planner 一样通过 sys.modules 解引用，以便 conftest 的 llm_stub 能接管。
    """

    llm_chat_mod = sys.modules["app.llm.chat"]
    res = llm_chat_mod._create_chat_completion(
        profile=PROFILE_ROUTING,
        trace_stage="multi_hop_decompose",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (res.choices[0].message.content or "").strip()


def decompose_query(
    *,
    rewritten_query: str,
    role: str,
) -> DecomposeResult:
    """把 rewritten_query 拆成 subqueries；失败一律降级单跳。

    调用方（multi_hop_node）只需检查 `result.degraded_to_single_hop`：
    - True  → 用 result.subqueries[0] 跑单跳 retrieval + answer；
    - False → 进入正常多跳 loop。
    """

    rewritten = (rewritten_query or "").strip()
    if not rewritten:
        return _fallback_single_hop(
            rewritten_query="",
            reason="empty_rewritten_query",
            error_code=DEGRADE_REASON_DECOMPOSE_FAILED,
        )

    system_prompt = build_decompose_system_prompt(MAX_SUBQUERIES)
    user_prompt = build_decompose_user_prompt(rewritten, role)

    try:
        raw = _call_decompose_llm(system_prompt, user_prompt)
    except LLMCallError as exc:
        log_warning(
            "multi_hop_decompose",
            "LLM call failed; degrade to single-hop",
            {
                "code": exc.code,
                "profile": exc.profile,
                "provider": exc.provider,
                "model": exc.model,
            },
        )
        return _fallback_single_hop(
            rewritten,
            reason=f"llm_call_failed:{exc.code}",
            error_code=DEGRADE_REASON_DECOMPOSE_FAILED,
        )

    try:
        subqueries = _parse_decompose_payload(raw)
    except DecomposeParseError as exc:
        log_warning(
            "multi_hop_decompose",
            "decompose schema invalid; degrade to single-hop",
            {"detail": str(exc), "response_preview": raw[:200]},
        )
        return _fallback_single_hop(
            rewritten,
            reason=str(exc),
            error_code=DEGRADE_REASON_DECOMPOSE_FAILED,
        )

    if not subqueries:
        # Decomposer 自己判定无法拆解（见 prompt 规则 1），走兜底。
        return _fallback_single_hop(
            rewritten,
            reason="llm_returned_empty_subqueries",
            error_code=DEGRADE_REASON_DECOMPOSE_FAILED,
        )

    if _is_synonym_degradation(subqueries, rewritten):
        return _fallback_single_hop(
            rewritten,
            reason="single_subquery_same_as_rewritten",
            error_code=DEGRADE_REASON_SYNONYM_SUBQUERY,
        )

    return DecomposeResult(
        subqueries=subqueries,
        degraded_to_single_hop=False,
        reason="",
        error_code="",
    )


__all__ = [
    "DecomposeParseError",
    "decompose_query",
]

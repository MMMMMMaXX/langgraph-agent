"""Multi-hop RAG 常量（Phase 3 PR-1）。

集中登记 multi-hop 路径上所有跨模块共享的上限、路由枚举、触发/负向门控正则，
以及 gap detector / decomposer 的阈值。下游模块（decomposer、gap detector、
multi_hop_node、supervisor/classifier、Composer、Verifier）只从此处导入，**禁止**
在各模块内维持副本（遵循 `feedback_constant_extraction.md`）。

触发规则的设计背景见 `docs/phase3-multi-hop-rag.md` §2.3：
- `MULTI_HOP_TRIGGERS`：跨文档链式 / 方案生成 / 依赖推导类正向信号；
- `MULTI_HOP_NEGATIVE_GATES`：简单定义 / 二元对比类反向信号，优先级最高，
  命中即强制回落单跳，避免无谓 decompose + 多段 retrieval 的延迟成本。
"""

from __future__ import annotations

from typing import Final

# ---- Hop / subquery / chunk 上限 ---------------------------------------

# 单请求内允许的最大 hop 数。超限即用已收集证据合成降级答案，而不是继续循环。
MAX_HOPS: Final[int] = 3

# Decomposer 单次允许产出的最大子查询数。超限直接判 schema 失败 → 回落单跳。
MAX_SUBQUERIES: Final[int] = 4

# 全局 chunk 上限，答案 LLM 前截断。保护 context / latency。
MAX_TOTAL_CHUNKS: Final[int] = 12

# ---- Supervisor / Classifier ---------------------------------------------

# query_classifier 的新分类值；仅用于 supervisor 路由判定，不进入 RAG 内部
# QUERY_TYPE_* 体系，避免污染单跳路径的策略表。
QUERY_CLASS_MULTI_HOP: Final[str] = "multi_hop"

# Supervisor 路由目标；与 `app/constants/routes.py` 的 ROUTE_* 风格一致，
# 导出点依旧集中在 routes.py（见 PR-3 会在 routes.py 里 re-export）。
ROUTE_MULTI_HOP_AGENT: Final[str] = "multi_hop_agent"

# ---- Composer / Verifier 风险码 ------------------------------------------

# RAG 单跳的 RISK_WARN_RAG_MISSING_CITATION 只管"doc_used=True 但无 citation"；
# multi-hop 追加场景：某个 subquery 压根没召回任何 chunk，覆盖度缺失。
# 文案由 `app/constants/workflow.py:RISK_WARN_LABELS` 承载，不在本模块重复。
RISK_WARN_MULTI_HOP_COVERAGE: Final[str] = "multi_hop_missing_coverage"

# ---- 触发 / 负向门控正则 -------------------------------------------------

# 正向触发：命中即进 multi-hop。
# 覆盖"基于 A 和 B 生成方案 / 结合 X 和 Y / 先查 X 再 Y / 根据 A 的指标分析 B /
# 跨项目跨系统"等跨文档链式推导语义。
# 注意：这些是"候选触发"，真正落地前必须先过 NEGATIVE_GATES——见 `_classify()`。
MULTI_HOP_TRIGGERS: Final[tuple[str, ...]] = (
    r"基于.*(和|以及).*?(生成|写|产出|起草|设计)",
    r"结合.*(和|以及).*?(写|生成|给出|产出|设计)",
    r"先.{0,10}(查|看|找).*再",
    r"根据.*的.*(分析|推导|评估)",
    r"跨(项目|系统|文档|部门|团队)",
)

# 负向门控：命中任一条即**强制单跳**，哪怕 TRIGGERS 也命中。
# 这里覆盖中文简单对比 / 定义类的常见写法。实现侧用 re.search（不是 fullmatch），
# 并在 gate 入口先做 strip + 半角化 + 末尾标点归一（? ？ → ""），
# 保证 "WAI-ARIA 和虚拟列表有什么区别？" / "X vs Y" / "X 对比 Y" / "什么是 JWT？" 全能拦下。
MULTI_HOP_NEGATIVE_GATES: Final[tuple[str, ...]] = (
    # "什么是 X" / "定义 X" / "介绍 X"
    r"^(什么是|定义|介绍).{0,40}$",
    # "X 是什么"
    r"^.{1,30}是什么$",
    # "X vs Y" —— 英文简写对比，后面不带 "区别" 也算简单对比
    r"^.{1,30}\s+vs\s+.{1,30}$",
    # 二元对比：X (和|与|vs|对比|比较) Y (有什么)?(区别|差异|不同) (是什么)?
    # 允许中间的空格，末尾问号在预处理里已去掉
    r"^.{1,30}\s*(和|与|vs|对比|比较)\s*.{1,30}\s*(有什么)?\s*(区别|差异|不同)(\s*是什么)?$",
    # "X 对比 Y" / "X 比较 Y" / "X 相比 Y"
    r"^.{1,30}\s*(对比|比较|相比)\s*.{1,30}$",
    # "X 和 Y 哪个更好/快/稳定/合适" 等
    r"^.{1,30}\s*(和|与)\s*.{1,30}\s*哪(个|种)(更)?(好|快|稳定|合适|强|差)$",
)

# ---- Gap detector 阈值 ---------------------------------------------------

# 每个 subquery 最少需要的 chunk 数。不足触发 `insufficient_chunks`。
MIN_CHUNKS_PER_SUBQUERY: Final[int] = 2

# 单 chunk 最低置信度。若所有 chunk 都低于此阈值触发 `low_confidence`。
MIN_CHUNK_SCORE: Final[float] = 0.3

# per_subquery_coverage 判定达标阈值（0-1），低于此值视为本 subquery 需要 refine。
PER_SUBQUERY_OK_THRESHOLD: Final[float] = 0.6

# global_coverage 判定达标阈值；仅对 comparison / 含 depends_on 链的 plan 启用。
GLOBAL_OK_THRESHOLD: Final[float] = 0.7

# 跨文档 comparison 场景下要求的最少来源文档数。
MIN_DOCS_MULTI: Final[int] = 2

# EvidencePreview.preview 上限（字节/字符），避免 state / trace 膨胀与全文泄漏。
EVIDENCE_PREVIEW_MAX_CHARS: Final[int] = 120

# ---- Decomposer 意图枚举 -------------------------------------------------

# Decomposer 输出的 intent 必须落在此集合；其他值视为 schema 非法 → 回落单跳。
SUBQUERY_INTENT_ENTITY_LOOKUP: Final[str] = "entity_lookup"
SUBQUERY_INTENT_PROCEDURE: Final[str] = "procedure"
SUBQUERY_INTENT_DEFINITION: Final[str] = "definition"
SUBQUERY_INTENT_COMPARISON_ARM: Final[str] = "comparison_arm"

VALID_SUBQUERY_INTENTS: Final[tuple[str, ...]] = (
    SUBQUERY_INTENT_ENTITY_LOOKUP,
    SUBQUERY_INTENT_PROCEDURE,
    SUBQUERY_INTENT_DEFINITION,
    SUBQUERY_INTENT_COMPARISON_ARM,
)

# ---- 降级原因 -----------------------------------------------------------

DEGRADE_REASON_DECOMPOSE_FAILED: Final[str] = "decompose_failed"
DEGRADE_REASON_SYNONYM_SUBQUERY: Final[str] = "synonym_subquery"
DEGRADE_REASON_BUDGET_EXCEEDED: Final[str] = "budget_exceeded"
DEGRADE_REASON_EVIDENCE_EMPTY: Final[str] = "evidence_empty"
DEGRADE_REASON_ANSWER_LLM_FAILED: Final[str] = "answer_llm_failed"

# PARTIAL 答案末尾追加的用户可见中文提示。Composer 的 `_append_degrade_notice`
# 按 degrade_reason 查此表；未登记的 reason 回退到通用提示，避免静默丢信号。
DEGRADE_NOTICE_LABELS: Final[dict[str, str]] = {
    DEGRADE_REASON_BUDGET_EXCEEDED: (
        "提示：受多跳预算限制，部分子问题可能未覆盖完整，答案仅基于已检索到的证据。"
    ),
    DEGRADE_REASON_DECOMPOSE_FAILED: (
        "提示：问题拆解失败，已回落到单跳检索，答案可能不如多跳推理全面。"
    ),
    DEGRADE_REASON_EVIDENCE_EMPTY: ("提示：未检索到足够证据，以下内容仅供参考。"),
    DEGRADE_REASON_ANSWER_LLM_FAILED: ("提示：答案生成过程异常，已尽力返回可用信息。"),
}
DEGRADE_NOTICE_FALLBACK: Final[str] = "提示：本次回答存在降级，内容可能不完整。"

# ---- step_results key ----------------------------------------------------

# multi_hop_node 把 pseudo-step 写入 `step_results[MULTI_HOP_STEP_ID]`；
# Composer 的直通分支、Verifier 的 coverage 检查、eval 断言都读同一个 key，
# 避免字面量"mh1"散落各处。
MULTI_HOP_STEP_ID: Final[str] = "mh1"


__all__ = [
    "DEGRADE_NOTICE_FALLBACK",
    "DEGRADE_NOTICE_LABELS",
    "DEGRADE_REASON_ANSWER_LLM_FAILED",
    "DEGRADE_REASON_BUDGET_EXCEEDED",
    "DEGRADE_REASON_DECOMPOSE_FAILED",
    "DEGRADE_REASON_EVIDENCE_EMPTY",
    "DEGRADE_REASON_SYNONYM_SUBQUERY",
    "EVIDENCE_PREVIEW_MAX_CHARS",
    "GLOBAL_OK_THRESHOLD",
    "MAX_HOPS",
    "MAX_SUBQUERIES",
    "MAX_TOTAL_CHUNKS",
    "MIN_CHUNKS_PER_SUBQUERY",
    "MIN_CHUNK_SCORE",
    "MIN_DOCS_MULTI",
    "MULTI_HOP_NEGATIVE_GATES",
    "MULTI_HOP_STEP_ID",
    "MULTI_HOP_TRIGGERS",
    "PER_SUBQUERY_OK_THRESHOLD",
    "QUERY_CLASS_MULTI_HOP",
    "RISK_WARN_MULTI_HOP_COVERAGE",
    "ROUTE_MULTI_HOP_AGENT",
    "SUBQUERY_INTENT_COMPARISON_ARM",
    "SUBQUERY_INTENT_DEFINITION",
    "SUBQUERY_INTENT_ENTITY_LOOKUP",
    "SUBQUERY_INTENT_PROCEDURE",
    "VALID_SUBQUERY_INTENTS",
]

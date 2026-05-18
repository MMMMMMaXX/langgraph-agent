"""可观测性 bucket / 基数 / 枚举常量（PR-0）。

与 `app/constants/metrics.py` 互补：
- metrics.py 管"指标契约"（metric 名、允许 label）；
- 本模块管"指标取值约束"（histogram bucket、label 基数上限、code 枚举）。

后续 PR 在埋点时只能选用本模块定义的 bucket / 枚举，不允许在调用处现写
裸字符串 / 裸 bucket，防止 dashboard / alert 因字面量漂移失效
（feedback_constant_extraction.md）。
"""

from __future__ import annotations

from typing import Final

# ---- Histogram bucket 边界 ----------------------------------------------
#
# 这些 bucket 必须显式声明并复用。`prometheus_client.Histogram` 默认 bucket
# 不适配本项目的 LLM/TTFT 量级，错用会导致 P99 永远落在 +Inf。

# 时间类（毫秒）：覆盖 5ms 内的快路径到 60s 的 LLM 慢请求。
HISTOGRAM_BUCKETS_DURATION_MS: Final[tuple[float, ...]] = (
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
    30000,
    60000,
)

# 数量类（条数 / 个数）：retrieval results、tool args 等。
HISTOGRAM_BUCKETS_RESULTS: Final[tuple[float, ...]] = (
    0,
    1,
    2,
    5,
    10,
    20,
    50,
    100,
)

# 字节类：covers 1KB ~ 1GB，主要用于 chunk size、payload size、Chroma segment。
HISTOGRAM_BUCKETS_BYTES: Final[tuple[float, ...]] = (
    1_024,
    10_240,
    102_400,
    1_048_576,
    10_485_760,
    104_857_600,
    1_073_741_824,
)


# ---- Label 基数上限 -----------------------------------------------------
#
# 单 metric × 单 label 累计取值数超过该硬上限即丢弃 + 自监控告警。
# 1000 来自监控方案 §4.2，数量级足够覆盖 tool_name / route_template / model
# 类合法 label，又能挡住 user_id / doc_id 这类高基数误用。
LABEL_CARDINALITY_HARD_LIMIT: Final[int] = 1000


# ---- Route template fallback -------------------------------------------
#
# FastAPI middleware 在请求未匹配到任何 route 时使用该 placeholder，
# 避免把 404 的真实 path 当成 route_template 写入 metrics。
ROUTE_TEMPLATE_UNMATCHED: Final[str] = "__no_route__"


# ---- Redaction placeholder ---------------------------------------------
#
# redactor 命中敏感片段时统一替换为该字符串。明确区别于"空字符串/缺失"，
# dashboard / log 检索时能直接 grep 该 token 找到所有脱敏点。
REDACTED_PLACEHOLDER: Final[str] = "<redacted>"


# ---- Trace / metric 内文本预览长度 -------------------------------------
#
# label 不允许放原始文本；少数确实需要进 trace metadata 的字段（如 chunk
# preview）必须截断到该长度，避免 sink 体积爆炸 + 全文泄漏。
TRACE_PREVIEW_MAX_CHARS: Final[int] = 120


__all__ = [
    "HISTOGRAM_BUCKETS_BYTES",
    "HISTOGRAM_BUCKETS_DURATION_MS",
    "HISTOGRAM_BUCKETS_RESULTS",
    "LABEL_CARDINALITY_HARD_LIMIT",
    "REDACTED_PLACEHOLDER",
    "ROUTE_TEMPLATE_UNMATCHED",
    "TRACE_PREVIEW_MAX_CHARS",
]

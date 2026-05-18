"""可观测性 metrics 名 / label / sink 注册表（PR-0）。

所有埋点必须先在本模块登记 metric 名 + 允许的 label 集合，业务代码里出现
未登记字符串的将被 lint / emit 包装器拒绝（feedback_constant_extraction.md）。

后续 PR (PR-1 ~ PR-7) 在落地具体指标时，向 `_REGISTRY` 追加 `MetricSpec`，
本模块负责单一事实来源 + 校验，不做埋点本身。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class MetricKind(str, Enum):
    """指标类型。值与 `prometheus_client` 子类小写名一致，方便 backend 选择。"""

    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"


class Sink(str, Enum):
    """可观测性 sink 类型，用于自监控失败统计。"""

    METRICS = "metrics"
    LOG = "log"
    TRACE = "trace"


class DropReason(str, Enum):
    """label 取值被 emit 包装器丢弃的原因枚举。

    `cardinality_overflow`：单 metric × 单 label 已积累的取值数超过基数上限。
    `not_whitelisted`：label 名不在该 metric 的允许 label 集合内。
    `redacted`：label 取值疑似敏感信息（API key / token / prompt 全文），
       被 redactor 拦下。
    """

    CARDINALITY_OVERFLOW = "cardinality_overflow"
    NOT_WHITELISTED = "not_whitelisted"
    REDACTED = "redacted"


# ---- Metric 名 ---------------------------------------------------------
#
# 命名约定（详见 docs/observability-monitoring-plan.md §4）：
# - counter 必须以 `_total` 结尾；
# - histogram 时间类用 `_duration_ms`，数量类用 `_results`，字节类用 `_bytes`；
# - 不允许把 user_id / request_id / 完整 query 作为 label。

# 自监控：emit / sink 写入失败计数（fire-and-forget 兜底统计）。
METRIC_OBSERVABILITY_EMIT_FAILED_TOTAL: Final[str] = "observability_emit_failed_total"

# 自监控：label 取值被丢弃计数（基数超限 / 未白名单 / 命中 redactor）。
METRIC_OBSERVABILITY_LABEL_DROPPED_TOTAL: Final[str] = (
    "observability_label_dropped_total"
)

# 自监控：Prometheus multiproc dir 文件数与 worker 数比值异常。
METRIC_OBSERVABILITY_MULTIPROC_DIR_ANOMALY_TOTAL: Final[str] = (
    "observability_multiproc_dir_anomaly_total"
)

# 服务层：HTTP 请求数（按 route_template / method / status_class 拆分）。
METRIC_HTTP_REQUEST_TOTAL: Final[str] = "http_request_total"

# 服务层：HTTP 请求耗时（毫秒）。
METRIC_HTTP_REQUEST_DURATION_MS: Final[str] = "http_request_duration_ms"

# 服务层：进程级版本指标，labels 仅承载 version / git_sha / build_time，值恒为 1。
METRIC_APP_VERSION_INFO: Final[str] = "app_version_info"


# ---- Label 名 ----------------------------------------------------------

LABEL_SINK: Final[str] = "sink"
LABEL_METRIC_NAME: Final[str] = "metric_name"
LABEL_LABEL_NAME: Final[str] = "label_name"
LABEL_DROP_REASON: Final[str] = "drop_reason"

# HTTP / 服务层 label 名。
LABEL_ROUTE_TEMPLATE: Final[str] = "route_template"
LABEL_METHOD: Final[str] = "method"
LABEL_STATUS_CLASS: Final[str] = "status_class"

# 版本元信息 label 名（与 Dockerfile 注入的环境变量一一对应）。
LABEL_VERSION: Final[str] = "version"
LABEL_GIT_SHA: Final[str] = "git_sha"
LABEL_BUILD_TIME: Final[str] = "build_time"


# ---- Spec 与注册表 -----------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """单个指标的契约：名字 / 类型 / 允许 label 集合 / 描述。

    后续 PR 不允许在业务代码里裸写 `Counter("foo", ...)`，必须通过本表登记后
    经 `app.observability.emit` 入口出指标。
    """

    name: str
    kind: MetricKind
    description: str
    labels: frozenset[str] = field(default_factory=frozenset)


# 内部可变注册表。后续 PR 通过 `register_metric` 追加，不直接修改 dict，
# 防止意外覆盖已登记的 spec。
_REGISTRY: dict[str, MetricSpec] = {}


def register_metric(spec: MetricSpec) -> None:
    """登记一个指标契约。重复登记同名 metric 视为冲突。"""

    existing = _REGISTRY.get(spec.name)
    if existing is not None and existing != spec:
        raise ValueError(
            f"metric '{spec.name}' already registered with a different spec"
        )
    _REGISTRY[spec.name] = spec


def get_metric_spec(name: str) -> MetricSpec | None:
    return _REGISTRY.get(name)


def all_metric_specs() -> tuple[MetricSpec, ...]:
    return tuple(_REGISTRY.values())


# ---- PR-0 自监控指标登记 ------------------------------------------------

register_metric(
    MetricSpec(
        name=METRIC_OBSERVABILITY_EMIT_FAILED_TOTAL,
        kind=MetricKind.COUNTER,
        description="metrics/log/trace sink 写入异常计数（fire-and-forget 自监控）。",
        labels=frozenset({LABEL_SINK}),
    )
)

register_metric(
    MetricSpec(
        name=METRIC_OBSERVABILITY_LABEL_DROPPED_TOTAL,
        kind=MetricKind.COUNTER,
        description=(
            "label 取值被 emit 包装器丢弃计数；按 (metric, label, drop_reason) 拆分，"
            "用于发现高基数泄漏与未登记字符串。"
        ),
        labels=frozenset({LABEL_METRIC_NAME, LABEL_LABEL_NAME, LABEL_DROP_REASON}),
    )
)


# ---- PR-1 服务层指标登记 ------------------------------------------------

register_metric(
    MetricSpec(
        name=METRIC_OBSERVABILITY_MULTIPROC_DIR_ANOMALY_TOTAL,
        kind=MetricKind.COUNTER,
        description=(
            "Prometheus multiproc dir 文件数与 worker 数比值异常计数；用于发现"
            "崩溃残留 / pid 复用 / 巡检清理事件。"
        ),
        labels=frozenset(),
    )
)

register_metric(
    MetricSpec(
        name=METRIC_HTTP_REQUEST_TOTAL,
        kind=MetricKind.COUNTER,
        description="HTTP 请求计数（按 route_template / method / status_class 拆分）。",
        labels=frozenset({LABEL_ROUTE_TEMPLATE, LABEL_METHOD, LABEL_STATUS_CLASS}),
    )
)

register_metric(
    MetricSpec(
        name=METRIC_HTTP_REQUEST_DURATION_MS,
        kind=MetricKind.HISTOGRAM,
        description="HTTP 请求耗时（毫秒），bucket 复用 HISTOGRAM_BUCKETS_DURATION_MS。",
        labels=frozenset({LABEL_ROUTE_TEMPLATE, LABEL_METHOD, LABEL_STATUS_CLASS}),
    )
)

register_metric(
    MetricSpec(
        name=METRIC_APP_VERSION_INFO,
        kind=MetricKind.GAUGE,
        description="进程版本元信息指标，值恒为 1，仅承载 version / git_sha / build_time。",
        labels=frozenset({LABEL_VERSION, LABEL_GIT_SHA, LABEL_BUILD_TIME}),
    )
)


__all__ = [
    "DropReason",
    "LABEL_BUILD_TIME",
    "LABEL_DROP_REASON",
    "LABEL_GIT_SHA",
    "LABEL_LABEL_NAME",
    "LABEL_METHOD",
    "LABEL_METRIC_NAME",
    "LABEL_ROUTE_TEMPLATE",
    "LABEL_SINK",
    "LABEL_STATUS_CLASS",
    "LABEL_VERSION",
    "METRIC_APP_VERSION_INFO",
    "METRIC_HTTP_REQUEST_DURATION_MS",
    "METRIC_HTTP_REQUEST_TOTAL",
    "METRIC_OBSERVABILITY_EMIT_FAILED_TOTAL",
    "METRIC_OBSERVABILITY_LABEL_DROPPED_TOTAL",
    "METRIC_OBSERVABILITY_MULTIPROC_DIR_ANOMALY_TOTAL",
    "MetricKind",
    "MetricSpec",
    "Sink",
    "all_metric_specs",
    "get_metric_spec",
    "register_metric",
]

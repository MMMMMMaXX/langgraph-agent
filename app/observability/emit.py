"""Fire-and-forget 埋点入口（PR-0）。

职责：
- 业务代码统一通过 `emit_counter / emit_histogram / emit_gauge` 出指标，
  禁止直接调用 backend；
- 校验 metric 名 / label 名是否登记；
- 校验 label 取值基数与白名单，命中限制即丢弃 + 自监控告警；
- backend 抛错全部吞掉，仅记录 `observability_emit_failed_total` 与本地日志。

PR-0 阶段 backend 默认是 `NullBackend`，只把样本累加到内存中，方便单测。
PR-1 会注入 prometheus_client 多进程 backend；本模块设计为后续 swap 不需要
改业务代码。
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Protocol

from app.constants.metrics import (
    LABEL_DROP_REASON,
    LABEL_LABEL_NAME,
    LABEL_METRIC_NAME,
    LABEL_SINK,
    METRIC_OBSERVABILITY_EMIT_FAILED_TOTAL,
    METRIC_OBSERVABILITY_LABEL_DROPPED_TOTAL,
    DropReason,
    MetricKind,
    MetricSpec,
    Sink,
    get_metric_spec,
)
from app.constants.observability import LABEL_CARDINALITY_HARD_LIMIT
from app.observability.redactor import is_banned_label_field, looks_sensitive
from app.utils.logger import logger


# ---- Backend 协议 ------------------------------------------------------


class MetricsBackend(Protocol):
    """metrics backend 抽象，PR-1 会用 prometheus_client 多进程实现替换。"""

    def counter_inc(
        self, name: str, labels: Mapping[str, str], value: float
    ) -> None: ...

    def histogram_observe(
        self, name: str, labels: Mapping[str, str], value: float
    ) -> None: ...

    def gauge_set(
        self, name: str, labels: Mapping[str, str], value: float
    ) -> None: ...


class NullBackend:
    """PR-0 默认 backend：内存累加，便于单测，无 IO 副作用。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self.histograms: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = {}
        self.gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}

    @staticmethod
    def _key(
        name: str, labels: Mapping[str, str]
    ) -> tuple[str, tuple[tuple[str, str], ...]]:
        return name, tuple(sorted(labels.items()))

    def counter_inc(
        self, name: str, labels: Mapping[str, str], value: float
    ) -> None:
        key = self._key(name, labels)
        with self._lock:
            self.counters[key] = self.counters.get(key, 0.0) + value

    def histogram_observe(
        self, name: str, labels: Mapping[str, str], value: float
    ) -> None:
        key = self._key(name, labels)
        with self._lock:
            self.histograms.setdefault(key, []).append(value)

    def gauge_set(
        self, name: str, labels: Mapping[str, str], value: float
    ) -> None:
        key = self._key(name, labels)
        with self._lock:
            self.gauges[key] = value

    def reset(self) -> None:
        """单测专用：清空累计样本。"""

        with self._lock:
            self.counters.clear()
            self.histograms.clear()
            self.gauges.clear()


# ---- 全局 backend 与基数簿 ---------------------------------------------

_backend: MetricsBackend = NullBackend()
_backend_lock = threading.Lock()

# 单 metric × 单 label 已记录的取值集合，用于 cardinality 兜底。
_cardinality_seen: dict[tuple[str, str], set[str]] = {}
_cardinality_lock = threading.Lock()


def set_backend(backend: MetricsBackend) -> None:
    """注入 backend。PR-1 在应用启动时调用一次。"""

    global _backend
    with _backend_lock:
        _backend = backend


def get_backend() -> MetricsBackend:
    return _backend


def reset_for_tests() -> None:
    """单测辅助：重置 backend（如果是 NullBackend）+ 基数簿。

    单元测试之间共享进程，必须显式清空，否则测试间会互相污染基数计数。
    """

    backend = _backend
    if isinstance(backend, NullBackend):
        backend.reset()
    with _cardinality_lock:
        _cardinality_seen.clear()


# ---- 内部：自监控指标 ---------------------------------------------------


def _emit_self_monitor(
    name: str, labels: Mapping[str, str], value: float = 1.0
) -> None:
    """直接调 backend 写自监控 counter，绕过校验避免递归。

    `observability_emit_failed_total` / `observability_label_dropped_total`
    自身命中校验失败时不会再触发自监控（fire-and-forget 三级降级里的最后一级
    交给本地日志兜底）。
    """

    try:
        _backend.counter_inc(name, labels, value)
    except Exception as exc:  # noqa: BLE001 — sink 永不抛回主链路
        logger.warning(
            {
                "event": "observability_emit_failed",
                "stage": "self_monitor",
                "metric": name,
                "error": str(exc),
            }
        )


def _record_emit_failure(sink: Sink, exc: BaseException) -> None:
    """sink 写入失败时的兜底：先尝试 self_monitor，再落本地日志。"""

    try:
        _emit_self_monitor(
            METRIC_OBSERVABILITY_EMIT_FAILED_TOTAL,
            {LABEL_SINK: sink.value},
        )
    finally:
        logger.warning(
            {
                "event": "observability_emit_failed",
                "sink": sink.value,
                "error": str(exc),
            }
        )


def _record_label_drop(metric: str, label: str, reason: DropReason) -> None:
    """label 取值被丢弃时上报自监控。"""

    _emit_self_monitor(
        METRIC_OBSERVABILITY_LABEL_DROPPED_TOTAL,
        {
            LABEL_METRIC_NAME: metric,
            LABEL_LABEL_NAME: label,
            LABEL_DROP_REASON: reason.value,
        },
    )


# ---- 内部：label 校验 ---------------------------------------------------


def _is_self_monitor_metric(name: str) -> bool:
    return name in (
        METRIC_OBSERVABILITY_EMIT_FAILED_TOTAL,
        METRIC_OBSERVABILITY_LABEL_DROPPED_TOTAL,
    )


def _validate_labels(
    spec: MetricSpec, labels: Mapping[str, str]
) -> dict[str, str]:
    """对 label dict 做白名单 + 基数 + redactor 三重校验。

    返回经过过滤后的 label dict。命中校验失败的 label 取值会被剔除并触发
    `observability_label_dropped_total`，**不会**让整个埋点失败。
    """

    cleaned: dict[str, str] = {}
    is_self_monitor = _is_self_monitor_metric(spec.name)

    for key, value in labels.items():
        # 1) 白名单：label 名必须登记
        if key not in spec.labels:
            if not is_self_monitor:
                _record_label_drop(spec.name, key, DropReason.NOT_WHITELISTED)
            continue

        text = "" if value is None else str(value)

        # 2) 字段名级别的禁用（user_id / query / prompt 等永不入 metric）
        if is_banned_label_field(key) and not is_self_monitor:
            _record_label_drop(spec.name, key, DropReason.NOT_WHITELISTED)
            continue

        # 3) redactor：取值疑似敏感串
        if looks_sensitive(text) and not is_self_monitor:
            _record_label_drop(spec.name, key, DropReason.REDACTED)
            continue

        # 4) 基数硬上限
        if not is_self_monitor:
            cardinality_key = (spec.name, key)
            with _cardinality_lock:
                seen = _cardinality_seen.setdefault(cardinality_key, set())
                if text not in seen:
                    if len(seen) >= LABEL_CARDINALITY_HARD_LIMIT:
                        # 释放锁前记录 drop，避免持锁触发 self_monitor 递归
                        overflow = True
                    else:
                        seen.add(text)
                        overflow = False
                else:
                    overflow = False
            if overflow:
                _record_label_drop(
                    spec.name, key, DropReason.CARDINALITY_OVERFLOW
                )
                continue

        cleaned[key] = text

    return cleaned


# ---- 公共 API ----------------------------------------------------------


def _emit(
    name: str,
    expected_kind: MetricKind,
    labels: Mapping[str, str] | None,
    value: float,
) -> None:
    spec = get_metric_spec(name)
    if spec is None:
        # 未登记的指标直接丢弃 + 自监控，不允许偷偷出指标。
        _record_label_drop(name, "__metric__", DropReason.NOT_WHITELISTED)
        return
    if spec.kind is not expected_kind:
        logger.warning(
            {
                "event": "observability_emit_kind_mismatch",
                "metric": name,
                "expected": expected_kind.value,
                "actual": spec.kind.value,
            }
        )
        return

    cleaned = _validate_labels(spec, labels or {})

    try:
        if expected_kind is MetricKind.COUNTER:
            _backend.counter_inc(name, cleaned, value)
        elif expected_kind is MetricKind.HISTOGRAM:
            _backend.histogram_observe(name, cleaned, value)
        else:
            _backend.gauge_set(name, cleaned, value)
    except Exception as exc:  # noqa: BLE001 — fire-and-forget
        _record_emit_failure(Sink.METRICS, exc)


def emit_counter(
    name: str, labels: Mapping[str, str] | None = None, value: float = 1.0
) -> None:
    """累加 counter。失败永不抛回主链路。"""

    _emit(name, MetricKind.COUNTER, labels, value)


def emit_histogram(
    name: str, value: float, labels: Mapping[str, str] | None = None
) -> None:
    """observe histogram 样本。失败永不抛回主链路。"""

    _emit(name, MetricKind.HISTOGRAM, labels, value)


def emit_gauge(
    name: str, value: float, labels: Mapping[str, str] | None = None
) -> None:
    """设置 gauge 当前值。失败永不抛回主链路。"""

    _emit(name, MetricKind.GAUGE, labels, value)


__all__ = [
    "MetricsBackend",
    "NullBackend",
    "emit_counter",
    "emit_gauge",
    "emit_histogram",
    "get_backend",
    "reset_for_tests",
    "set_backend",
]

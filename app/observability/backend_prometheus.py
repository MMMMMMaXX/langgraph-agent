"""prometheus_client metrics backend（PR-1）。

职责：
- 把 `app/observability/emit.py` 的 backend 协议适配到 `prometheus_client`；
- 按 `app/constants/metrics.py:_REGISTRY` 的 spec 懒加载创建 Counter / Histogram / Gauge；
- 多 worker 安全：通过 `PROMETHEUS_MULTIPROC_DIR` 环境变量启用多进程模式，
  `/metrics` 端点使用 `MultiProcessCollector` 聚合各 worker 的样本文件；
- bucket 选择：histogram 名以 `_duration_ms` / `_results` / `_bytes` 结尾时
  从 `app/constants/observability.py` 取约定 bucket，避免 prometheus_client
  默认 bucket 让 P99 永远落在 +Inf。
"""

from __future__ import annotations

import threading
from collections.abc import Mapping

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

from app.constants.metrics import (
    MetricKind,
    MetricSpec,
    get_metric_spec,
)
from app.constants.observability import (
    HISTOGRAM_BUCKETS_BYTES,
    HISTOGRAM_BUCKETS_DURATION_MS,
    HISTOGRAM_BUCKETS_RESULTS,
)
from app.utils.logger import logger


def _bucket_for(metric_name: str) -> tuple[float, ...]:
    if metric_name.endswith("_duration_ms"):
        return HISTOGRAM_BUCKETS_DURATION_MS
    if metric_name.endswith("_bytes"):
        return HISTOGRAM_BUCKETS_BYTES
    return HISTOGRAM_BUCKETS_RESULTS


def _fill_labels(spec: MetricSpec, labels: Mapping[str, str]) -> dict[str, str]:
    """补齐 spec 要求的全部 label，缺失项填空串。

    prometheus_client 要求 `.labels(...)` 必须提供 spec 声明的全部 labelnames，
    emit 包装器可能会 drop 掉一部分 label 取值，此处统一兜底，避免崩 backend。
    """

    return {key: str(labels.get(key, "")) for key in spec.labels}


class PrometheusBackend:
    """生产 backend：把样本写入 prometheus_client。

    - 进程内单例，懒加载 metric 对象；
    - 多进程模式由 `PROMETHEUS_MULTIPROC_DIR` 环境变量隐式控制，
      prometheus_client 会自行选择按 pid 拆分的样本文件；
    - 失败时不抛回主链路，emit 包装器会记 `observability_emit_failed_total`。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._gauges: dict[str, Gauge] = {}

    # ---- 懒加载工厂 ----------------------------------------------------

    def _get_counter(self, spec: MetricSpec) -> Counter:
        with self._lock:
            metric = self._counters.get(spec.name)
            if metric is None:
                metric = Counter(
                    spec.name,
                    spec.description,
                    labelnames=sorted(spec.labels),
                )
                self._counters[spec.name] = metric
            return metric

    def _get_histogram(self, spec: MetricSpec) -> Histogram:
        with self._lock:
            metric = self._histograms.get(spec.name)
            if metric is None:
                metric = Histogram(
                    spec.name,
                    spec.description,
                    labelnames=sorted(spec.labels),
                    buckets=_bucket_for(spec.name),
                )
                self._histograms[spec.name] = metric
            return metric

    def _get_gauge(self, spec: MetricSpec) -> Gauge:
        with self._lock:
            metric = self._gauges.get(spec.name)
            if metric is None:
                # multiprocess 模式下 Gauge 必须显式指定聚合方式，否则各 worker
                # 读出来都是空。`mostrecent` 适合"最近一次值"，最贴近 dashboard 期望。
                metric = Gauge(
                    spec.name,
                    spec.description,
                    labelnames=sorted(spec.labels),
                    multiprocess_mode="mostrecent",
                )
                self._gauges[spec.name] = metric
            return metric

    # ---- MetricsBackend 协议实现 ---------------------------------------

    def counter_inc(
        self, name: str, labels: Mapping[str, str], value: float
    ) -> None:
        spec = get_metric_spec(name)
        if spec is None or spec.kind is not MetricKind.COUNTER:
            return
        metric = self._get_counter(spec)
        if spec.labels:
            metric.labels(**_fill_labels(spec, labels)).inc(value)
        else:
            metric.inc(value)

    def histogram_observe(
        self, name: str, labels: Mapping[str, str], value: float
    ) -> None:
        spec = get_metric_spec(name)
        if spec is None or spec.kind is not MetricKind.HISTOGRAM:
            return
        metric = self._get_histogram(spec)
        if spec.labels:
            metric.labels(**_fill_labels(spec, labels)).observe(value)
        else:
            metric.observe(value)

    def gauge_set(
        self, name: str, labels: Mapping[str, str], value: float
    ) -> None:
        spec = get_metric_spec(name)
        if spec is None or spec.kind is not MetricKind.GAUGE:
            return
        metric = self._get_gauge(spec)
        if spec.labels:
            metric.labels(**_fill_labels(spec, labels)).set(value)
        else:
            metric.set(value)


def render_exposition() -> tuple[bytes, str]:
    """生成 `/metrics` 端点响应。

    在多进程模式下使用 `MultiProcessCollector` 聚合 `$PROMETHEUS_MULTIPROC_DIR`
    下所有 worker 的样本；单进程模式下走默认 registry。
    """

    import os

    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        try:
            multiprocess.MultiProcessCollector(registry)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                {
                    "event": "observability_emit_failed",
                    "stage": "multiprocess_collector",
                    "error": str(exc),
                }
            )
            from prometheus_client import REGISTRY  # 兜底

            registry = REGISTRY
    else:
        from prometheus_client import REGISTRY

        registry = REGISTRY

    return generate_latest(registry), "text/plain; version=0.0.4; charset=utf-8"


__all__ = ["PrometheusBackend", "render_exposition"]

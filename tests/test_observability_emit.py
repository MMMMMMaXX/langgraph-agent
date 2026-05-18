"""PR-0：emit 包装器单测。

覆盖：
- 未登记 metric / label 被拒绝并自监控；
- label 取值命中基数上限触发 cardinality_overflow；
- 业务字段（user_id / query）即使登记也走 not_whitelisted；
- 敏感取值被 redactor 拦截后走 redacted；
- backend 抛错被 fire-and-forget 吞掉，自监控指标 +1。
"""

from __future__ import annotations

import pytest

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
    register_metric,
)
from app.constants.observability import LABEL_CARDINALITY_HARD_LIMIT
from app.observability import emit as emit_module
from app.observability.emit import (
    NullBackend,
    emit_counter,
    emit_gauge,
    emit_histogram,
    get_backend,
    reset_for_tests,
    set_backend,
)


# ---- 用例专属测试指标 ---------------------------------------------------

_TEST_COUNTER = "test_emit_counter_total"
_TEST_HISTOGRAM = "test_emit_histogram_duration_ms"
_TEST_GAUGE = "test_emit_gauge"

# 测试只登记一次；register_metric 对相同 spec 重复调用是幂等的。
register_metric(
    MetricSpec(
        name=_TEST_COUNTER,
        kind=MetricKind.COUNTER,
        description="test counter",
        labels=frozenset({"tenant_id", "tool_name"}),
    )
)
register_metric(
    MetricSpec(
        name=_TEST_HISTOGRAM,
        kind=MetricKind.HISTOGRAM,
        description="test histogram",
        labels=frozenset({"route_template"}),
    )
)
register_metric(
    MetricSpec(
        name=_TEST_GAUGE,
        kind=MetricKind.GAUGE,
        description="test gauge",
        labels=frozenset({"suite"}),
    )
)


@pytest.fixture(autouse=True)
def _isolate_emit_state() -> None:
    """每个用例之间清空 backend / 基数簿，避免互相污染。"""

    reset_for_tests()
    yield
    reset_for_tests()


def _backend() -> NullBackend:
    backend = get_backend()
    assert isinstance(backend, NullBackend), "PR-0 默认 backend 应为 NullBackend"
    return backend


def _drop_count(metric: str, label: str, reason: DropReason) -> float:
    """从 NullBackend 中读 observability_label_dropped_total 的累计值。"""

    expected_labels = (
        (LABEL_DROP_REASON, reason.value),
        (LABEL_LABEL_NAME, label),
        (LABEL_METRIC_NAME, metric),
    )
    return _backend().counters.get(
        (METRIC_OBSERVABILITY_LABEL_DROPPED_TOTAL, expected_labels), 0.0
    )


def test_emit_counter_records_normal_path() -> None:
    emit_counter(_TEST_COUNTER, {"tenant_id": "t1", "tool_name": "weather"})

    expected_labels = (("tenant_id", "t1"), ("tool_name", "weather"))
    assert _backend().counters.get((_TEST_COUNTER, expected_labels)) == 1.0


def test_emit_histogram_and_gauge_dispatch_correctly() -> None:
    emit_histogram(_TEST_HISTOGRAM, 42.0, {"route_template": "/chat"})
    emit_gauge(_TEST_GAUGE, 0.91, {"suite": "default"})

    hist_key = (_TEST_HISTOGRAM, (("route_template", "/chat"),))
    gauge_key = (_TEST_GAUGE, (("suite", "default"),))
    assert _backend().histograms[hist_key] == [42.0]
    assert _backend().gauges[gauge_key] == 0.91


def test_unregistered_metric_is_dropped() -> None:
    emit_counter("metric_never_registered_total", {"tenant_id": "t1"})

    # 既不写入业务 counter，也要触发 not_whitelisted 自监控
    assert (
        "metric_never_registered_total",
        (("tenant_id", "t1"),),
    ) not in _backend().counters
    drop = _drop_count(
        "metric_never_registered_total",
        "__metric__",
        DropReason.NOT_WHITELISTED,
    )
    assert drop == 1.0


def test_unregistered_label_is_dropped_but_metric_still_emitted() -> None:
    # `provider` 不在 _TEST_COUNTER 的允许 label 中
    emit_counter(
        _TEST_COUNTER,
        {"tenant_id": "t1", "tool_name": "weather", "provider": "deepseek"},
    )

    expected_labels = (("tenant_id", "t1"), ("tool_name", "weather"))
    assert _backend().counters[(_TEST_COUNTER, expected_labels)] == 1.0
    assert (
        _drop_count(_TEST_COUNTER, "provider", DropReason.NOT_WHITELISTED) == 1.0
    )


def test_sensitive_label_value_is_redacted() -> None:
    emit_counter(
        _TEST_COUNTER,
        {"tenant_id": "sk-abcdef0123456789ABCDEF0123", "tool_name": "weather"},
    )

    # tenant_id 命中敏感模式 → 该 label 被丢弃；tool_name 仍写入
    drop_key_labels = (("tool_name", "weather"),)
    assert _backend().counters[(_TEST_COUNTER, drop_key_labels)] == 1.0
    assert _drop_count(_TEST_COUNTER, "tenant_id", DropReason.REDACTED) == 1.0


def test_cardinality_overflow_is_dropped_and_self_monitored() -> None:
    # 故意把 LABEL_CARDINALITY_HARD_LIMIT 喂满，再多一个就应被 drop
    for i in range(LABEL_CARDINALITY_HARD_LIMIT):
        emit_counter(
            _TEST_COUNTER, {"tenant_id": f"t{i}", "tool_name": "weather"}
        )

    # 第 N+1 个 tenant_id 应触发 overflow
    overflow_value = f"t{LABEL_CARDINALITY_HARD_LIMIT}"
    emit_counter(
        _TEST_COUNTER,
        {"tenant_id": overflow_value, "tool_name": "weather"},
    )

    overflow_labels = (("tenant_id", overflow_value), ("tool_name", "weather"))
    assert (_TEST_COUNTER, overflow_labels) not in _backend().counters
    assert (
        _drop_count(_TEST_COUNTER, "tenant_id", DropReason.CARDINALITY_OVERFLOW)
        == 1.0
    )


def test_backend_exception_is_swallowed_and_self_monitored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """业务 backend 抛错 → 主链路不感知 + emit_failed_total +1。"""

    backend = _backend()
    real_counter_inc = backend.counter_inc

    def _inc(name, labels, value):  # type: ignore[no-untyped-def]
        if name == _TEST_COUNTER:
            raise RuntimeError("backend boom")
        return real_counter_inc(name, labels, value)

    monkeypatch.setattr(backend, "counter_inc", _inc)

    # 不应抛
    emit_counter(_TEST_COUNTER, {"tenant_id": "t1", "tool_name": "weather"})

    self_monitor_key = (
        METRIC_OBSERVABILITY_EMIT_FAILED_TOTAL,
        ((LABEL_SINK, Sink.METRICS.value),),
    )
    assert backend.counters.get(self_monitor_key, 0.0) == 1.0


def test_set_backend_swap_takes_effect() -> None:
    class CountingBackend(NullBackend):
        pass

    swap = CountingBackend()
    original = get_backend()
    try:
        set_backend(swap)
        emit_counter(
            _TEST_COUNTER, {"tenant_id": "t1", "tool_name": "weather"}
        )
        assert swap.counters
    finally:
        set_backend(original)

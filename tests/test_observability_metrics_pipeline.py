"""PR-1：HTTP middleware + /metrics endpoint + multiproc 工具集成测试。

通过 FastAPI TestClient 跑端到端：
- 中间件能记下 `http_request_total` / `http_request_duration_ms`，且 label 用
  `route_template` 而非真实路径；
- `/metrics` 在单进程模式下正常返回 Prometheus exposition；
- multiproc + reload 同时启用时 `assert_multiproc_safe` fail-fast；
- multiproc dir 启动前清理与 stale pid 巡检均能识别 pid 死亡的 db 文件。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.constants.metrics import (
    LABEL_METHOD,
    LABEL_ROUTE_TEMPLATE,
    LABEL_STATUS_CLASS,
    METRIC_HTTP_REQUEST_DURATION_MS,
    METRIC_HTTP_REQUEST_TOTAL,
    METRIC_OBSERVABILITY_MULTIPROC_DIR_ANOMALY_TOTAL,
)
from app.constants.observability import ROUTE_TEMPLATE_UNMATCHED
from app.observability.backend_prometheus import PrometheusBackend
from app.observability.emit import (
    NullBackend,
    get_backend,
    reset_for_tests,
    set_backend,
)
from app.observability.multiprocess import (
    ENV_OBS_ALLOW_RELOAD_MULTIPROC,
    ENV_OBS_LIFESPAN_PREPARE_MULTIPROC_DIR,
    ENV_PROMETHEUS_MULTIPROC_DIR,
    StalePidScanner,
    assert_multiproc_safe,
    is_reload_invocation,
    lifespan_should_prepare_multiproc_dir,
    prepare_multiproc_dir,
    scan_stale_pids,
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """每个用例使用全新 NullBackend，断言 emit 调用而不依赖 Prometheus 全局状态。

    `app.lifespan` 默认会注入 PrometheusBackend；这里在 TestClient 进入 lifespan 后
    再覆写为 NullBackend，方便直接读 backend 内部计数。
    """

    with TestClient(app) as c:
        original = get_backend()
        set_backend(NullBackend())
        reset_for_tests()
        try:
            yield c
        finally:
            set_backend(original)


def _backend() -> NullBackend:
    backend = get_backend()
    assert isinstance(backend, NullBackend)
    return backend


def _http_counter_value(route: str, method: str, status_class: str) -> float:
    expected_labels = (
        (LABEL_METHOD, method),
        (LABEL_ROUTE_TEMPLATE, route),
        (LABEL_STATUS_CLASS, status_class),
    )
    return _backend().counters.get((METRIC_HTTP_REQUEST_TOTAL, expected_labels), 0.0)


def test_middleware_records_health_request_with_route_template(
    client: TestClient,
) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200

    assert _http_counter_value("/health", "GET", "2xx") == 1.0
    # 同 label 集合下的 histogram 应该至少记到一笔耗时
    histograms = _backend().histograms
    hist_key = (
        METRIC_HTTP_REQUEST_DURATION_MS,
        (
            (LABEL_METHOD, "GET"),
            (LABEL_ROUTE_TEMPLATE, "/health"),
            (LABEL_STATUS_CLASS, "2xx"),
        ),
    )
    assert hist_key in histograms
    assert histograms[hist_key]
    assert histograms[hist_key][0] >= 0


def test_middleware_uses_unmatched_placeholder_for_unknown_path(
    client: TestClient,
) -> None:
    resp = client.get("/this-route-does-not-exist")
    assert resp.status_code == 404
    # 关键断言：route_template 不能是原始 URL，必须是固定 placeholder，
    # 否则 404 流量会把 path 维度打成高基数。
    assert _http_counter_value(ROUTE_TEMPLATE_UNMATCHED, "GET", "4xx") == 1.0
    assert _http_counter_value("/this-route-does-not-exist", "GET", "4xx") == 0.0


def test_metrics_endpoint_renders_prometheus_exposition() -> None:
    """/metrics 在单进程模式下应输出 Prometheus 文本格式。"""

    # 单独构造 client，让 lifespan 注入 PrometheusBackend
    with TestClient(app) as c:
        # 先发一笔已知请求，确保 /metrics 中能看到 http_request_total
        c.get("/health")
        resp = c.get("/metrics")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    body = resp.text
    assert "http_request_total" in body
    assert "app_version_info" in body


def test_assert_multiproc_safe_failfast_under_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(ENV_PROMETHEUS_MULTIPROC_DIR, str(tmp_path))
    monkeypatch.setattr(
        "app.observability.multiprocess.is_reload_invocation",
        lambda: True,
    )
    with pytest.raises(RuntimeError, match="reload"):
        assert_multiproc_safe()


def test_assert_multiproc_safe_allows_override_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """OBS_ALLOW_RELOAD_MULTIPROC=1 时，is_reload_invocation 应返回 False。"""

    monkeypatch.setenv(ENV_PROMETHEUS_MULTIPROC_DIR, str(tmp_path))
    monkeypatch.setattr("sys.argv", ["uvicorn", "app.api:app", "--reload"])
    monkeypatch.setenv(ENV_OBS_ALLOW_RELOAD_MULTIPROC, "1")

    assert is_reload_invocation() is False
    # 不应抛
    assert_multiproc_safe()


def test_assert_multiproc_safe_noop_without_multiproc_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_PROMETHEUS_MULTIPROC_DIR, raising=False)
    monkeypatch.setattr("sys.argv", ["uvicorn", "app.api:app", "--reload"])
    # 单进程模式下 reload 完全合法
    assert_multiproc_safe()


def test_prepare_multiproc_dir_clears_residual_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(ENV_PROMETHEUS_MULTIPROC_DIR, str(tmp_path))
    residual = tmp_path / "counter_99999.db"
    residual.write_bytes(b"stale")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "trash").write_bytes(b"x")

    prepare_multiproc_dir()

    assert tmp_path.exists()
    assert not residual.exists()
    assert not nested.exists()


def test_lifespan_should_prepare_multiproc_dir_default_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认关闭：多 worker 场景下若每个 lifespan 都清，会擦掉别的 worker 的样本。"""

    monkeypatch.delenv(ENV_OBS_LIFESPAN_PREPARE_MULTIPROC_DIR, raising=False)
    assert lifespan_should_prepare_multiproc_dir() is False


def test_lifespan_should_prepare_multiproc_dir_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式 OBS_LIFESPAN_PREPARE_MULTIPROC_DIR=1 才允许 lifespan 兜底清理。"""

    monkeypatch.setenv(ENV_OBS_LIFESPAN_PREPARE_MULTIPROC_DIR, "1")
    assert lifespan_should_prepare_multiproc_dir() is True


def test_lifespan_does_not_clear_multiproc_dir_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """启动 app 时不应自动擦掉 multiproc dir 里其它 worker 的文件。"""

    monkeypatch.setenv(ENV_PROMETHEUS_MULTIPROC_DIR, str(tmp_path))
    monkeypatch.delenv(ENV_OBS_LIFESPAN_PREPARE_MULTIPROC_DIR, raising=False)
    other_worker_file = tmp_path / "counter_88888.db"
    other_worker_file.write_bytes(b"belongs-to-another-worker")

    with TestClient(app):
        pass

    assert other_worker_file.exists()


def test_scan_stale_pids_cleans_dead_pid_db(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """只清 live gauge 文件；counter / histogram 即使 pid 死了也保留。"""

    monkeypatch.setenv(ENV_PROMETHEUS_MULTIPROC_DIR, str(tmp_path))

    # 已死 pid 的 live gauge 文件——应被清理
    dead_live = tmp_path / "gauge_livesum_99999.db"
    dead_live.write_bytes(b"dead-live")
    # 已死 pid 的 counter 文件——必须保留（累计样本不可丢）
    dead_counter = tmp_path / "counter_99999.db"
    dead_counter.write_bytes(b"dead-counter")
    # 活 pid 的 live gauge 文件——不应触发清理
    alive_live = tmp_path / f"gauge_livesum_{os.getpid()}.db"
    alive_live.write_bytes(b"alive")

    set_backend(NullBackend())
    reset_for_tests()
    cleaned = scan_stale_pids()
    assert cleaned == 1

    backend = _backend()
    anomaly_value = backend.counters.get(
        (METRIC_OBSERVABILITY_MULTIPROC_DIR_ANOMALY_TOTAL, ()), 0.0
    )
    assert anomaly_value == 1.0
    # counter 必须保留——这是 prometheus_client 的设计意图
    assert dead_counter.exists()
    # alive 进程的文件不能被动
    assert alive_live.exists()


def test_scan_stale_pids_dedupes_via_reported_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`reported` 集合可避免重复告警，即使文件没被真正删除。"""

    monkeypatch.setenv(ENV_PROMETHEUS_MULTIPROC_DIR, str(tmp_path))
    # 模拟 mark_process_dead 失败（文件仍残留）：直接 patch 掉删除动作
    monkeypatch.setattr(
        "app.observability.multiprocess.mark_process_dead_safe",
        lambda pid: None,
    )

    dead_live = tmp_path / "gauge_livesum_99999.db"
    dead_live.write_bytes(b"dead-live")

    set_backend(NullBackend())
    reset_for_tests()
    reported: set[int] = set()

    assert scan_stale_pids(reported=reported) == 1
    # 第二轮：同一个 dead pid 不应再被计入
    assert scan_stale_pids(reported=reported) == 0

    backend = _backend()
    anomaly_value = backend.counters.get(
        (METRIC_OBSERVABILITY_MULTIPROC_DIR_ANOMALY_TOTAL, ()), 0.0
    )
    assert anomaly_value == 1.0


def test_scan_stale_pids_noop_without_multiproc_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_PROMETHEUS_MULTIPROC_DIR, raising=False)
    assert scan_stale_pids() == 0


def test_stale_pid_scanner_does_not_start_without_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_PROMETHEUS_MULTIPROC_DIR, raising=False)
    scanner = StalePidScanner(interval_seconds=0.01)
    scanner.start()
    try:
        assert scanner._thread is None
    finally:
        scanner.stop()


def test_prometheus_backend_drops_unregistered_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend 拒绝未登记 metric，与 emit 的 not_whitelisted 一致。"""

    backend = PrometheusBackend()
    # 不应抛错，也不应创建 Counter
    backend.counter_inc("not_registered_total", {"foo": "bar"}, 1.0)
    assert "not_registered_total" not in backend._counters

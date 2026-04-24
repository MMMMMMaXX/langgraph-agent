"""覆盖 /health 和 /health/ready 的静态检查。

这些路由不会触发 LLM 调用，不需要 llm_stub。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # 即使构建 args 未注入，也应提供三个版本字段（默认值）
    assert {"version", "git_sha", "build_time"} <= set(body.keys())


def test_health_reflects_injected_version(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    monkeypatch.setenv("APP_VERSION", "v9.9.9")
    monkeypatch.setenv("APP_GIT_SHA", "deadbee")
    monkeypatch.setenv("APP_BUILD_TIME", "2026-04-23T00:00:00Z")

    resp = client.get("/health")
    body = resp.json()
    assert body["version"] == "v9.9.9"
    assert body["git_sha"] == "deadbee"
    assert body["build_time"] == "2026-04-23T00:00:00Z"


def test_health_ready_ok(client: TestClient) -> None:
    resp = client.get("/health/ready")
    # 本地 / CI 默认配置都应 ready
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    for name in ("llm_profile", "history_path", "chroma_dir"):
        assert body["checks"][name]["ok"] is True


def test_health_ready_fails_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    # 清空 DeepSeek key + 老的兼容字段，让 llm_profile 检查失败
    for key in ("DEEPSEEK_API_KEY", "API_KEY"):
        monkeypatch.setenv(key, "")

    resp = client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["llm_profile"]["ok"] is False
    # 其他两项不受影响
    assert body["checks"]["history_path"]["ok"] is True
    assert body["checks"]["chroma_dir"]["ok"] is True

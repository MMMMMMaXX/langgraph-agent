"""HTTP 请求埋点中间件（PR-1）。

职责：在每个 FastAPI 请求结束后记录 `http_request_total` /
`http_request_duration_ms`。

关键点（监控方案 §3.1 / §4.2）：
- label 必须用 `route_template`（FastAPI 注册的路径模板，例如
  `/knowledge/docs/{doc_id}/reindex`），**禁止**直接使用包含路径参数的原始 URL，
  否则 `doc_id` 会把 path 维度打成高基数；
- 状态码做桶化（`2xx / 4xx / 5xx`）作为 label，原始 status code 仅写日志，
  保持基数稳定；
- 失败完全 fire-and-forget，sink 异常永不抛回主链路，由 emit 包装器自监控。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

from app.constants.metrics import (
    LABEL_METHOD,
    LABEL_ROUTE_TEMPLATE,
    LABEL_STATUS_CLASS,
    METRIC_HTTP_REQUEST_DURATION_MS,
    METRIC_HTTP_REQUEST_TOTAL,
)
from app.constants.observability import ROUTE_TEMPLATE_UNMATCHED
from app.observability.emit import emit_counter, emit_histogram


def _extract_route_template(request: Request) -> str:
    """从 ASGI scope 中取 FastAPI 路由模板，未匹配走 placeholder。

    `request.scope["route"]` 在路由命中后由 FastAPI 写入；404 / 未注册路径
    没有该 key，必须 fallback 到固定字符串而不是 raw path，避免高基数。
    """

    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if isinstance(template, str) and template:
        return template
    return ROUTE_TEMPLATE_UNMATCHED


def _status_class(status_code: int) -> str:
    if status_code < 200:
        return "1xx"
    if status_code < 300:
        return "2xx"
    if status_code < 400:
        return "3xx"
    if status_code < 500:
        return "4xx"
    return "5xx"


async def _metrics_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        # 主链路异常仍按 5xx 计入指标，再原样抛回让 FastAPI 错误处理接管。
        status_code = 500
        raise
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        labels = {
            LABEL_ROUTE_TEMPLATE: _extract_route_template(request),
            LABEL_METHOD: request.method.upper(),
            LABEL_STATUS_CLASS: _status_class(status_code),
        }
        emit_counter(METRIC_HTTP_REQUEST_TOTAL, labels)
        emit_histogram(METRIC_HTTP_REQUEST_DURATION_MS, duration_ms, labels)


def install_metrics_middleware(app: FastAPI) -> None:
    """把 metrics 中间件挂到 FastAPI 应用上。"""

    app.middleware("http")(_metrics_middleware)


__all__ = ["install_metrics_middleware"]

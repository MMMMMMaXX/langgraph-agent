"""monitor.query_errors：read_only mock tool（Phase 2 workflow demo 用）。

为什么单独放 monitor.py 而不是追加进 tools.py：
- tools.py 汇聚的是 Phase 1 就存在的 weather/calculate 两个 demo 工具，结构紧凑。
- monitor 定位是"企业排障工具"，未来可能扩展 `monitor.query_slow_queries` /
  `monitor.query_traces` 等同系工具，单文件承载业务域更清晰；也便于未来把 mock
  数据替换成真实 APM 接入时不污染 tools.py。

安全属性：纯读、低风险、不需要 confirmation / idempotency。与
`app/tools/metadata.py:TOOL_METADATA` 的声明保持一致（常量抽离原则：具体属性写
在 metadata 那一份，这里只负责业务逻辑）。
"""

from __future__ import annotations

from app.constants.tooling import (
    MONITOR_QUERY_ERRORS_BY_SERVICE,
    MONITOR_QUERY_ERRORS_DEFAULT,
    MONITOR_SERVICE_MAX_CHARS,
)
from app.utils.logger import log_warning


def monitor_query_errors(service: str) -> str:
    """查询 service 最近 30 分钟的 5xx/错误摘要。

    - 空 / 过长输入直接返回兜底文案并记一条 warning；不抛，保持和
      `calculate` / `get_weather` 一致的"工具层自己兜底"风格，避免上层 agent
      因为一个 demo 工具崩溃。
    - 大小写敏感：service 名一般是稳定 slug（`payment-service`），不做智能
      归一化，减少歧义。
    """

    svc = (service or "").strip()
    if not svc:
        log_warning(
            "monitor_query_errors",
            "empty service",
            {"service_preview": ""},
        )
        return MONITOR_QUERY_ERRORS_DEFAULT
    if len(svc) > MONITOR_SERVICE_MAX_CHARS:
        log_warning(
            "monitor_query_errors",
            "service name too long",
            {"service_preview": svc[:MONITOR_SERVICE_MAX_CHARS]},
        )
        return MONITOR_QUERY_ERRORS_DEFAULT

    return MONITOR_QUERY_ERRORS_BY_SERVICE.get(svc, MONITOR_QUERY_ERRORS_DEFAULT)


__all__ = ["monitor_query_errors"]

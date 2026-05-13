"""Tool Safety 相关常量。

集中点：工具执行状态、风险等级、错误码、默认超时等跨模块共享的字符串/
数值都放在这里。`app/constants/tooling.py` 里的 TOOL_NAME_* 已经稳定，
保持原位不动；本文件只加新东西，避免改动波及既有 import。

本文件被 `app/tools/metadata.py`、`app/tools/execution_record.py`（PR 3）、
`app/tools/confirmation.py`（PR 4）、tool_agent 共同消费。
"""

from __future__ import annotations

from typing import Final

# -------- tool_executions.status 枚举 --------
# 设计文档 §2.4：pending / succeeded / failed / timeout_unknown。

TOOL_STATUS_PENDING: Final[str] = "pending"
TOOL_STATUS_SUCCEEDED: Final[str] = "succeeded"
TOOL_STATUS_FAILED: Final[str] = "failed"
# 本地 asyncio.wait_for 超时：下游是否执行未知，禁止自动重放，
# 交人工 reconcile / 下游幂等查询。
TOOL_STATUS_TIMEOUT_UNKNOWN: Final[str] = "timeout_unknown"

VALID_TOOL_STATUSES: Final[frozenset[str]] = frozenset(
    {
        TOOL_STATUS_PENDING,
        TOOL_STATUS_SUCCEEDED,
        TOOL_STATUS_FAILED,
        TOOL_STATUS_TIMEOUT_UNKNOWN,
    }
)

# -------- 风险等级 --------

RISK_LEVEL_LOW: Final[str] = "low"
RISK_LEVEL_MEDIUM: Final[str] = "medium"
RISK_LEVEL_HIGH: Final[str] = "high"

VALID_RISK_LEVELS: Final[frozenset[str]] = frozenset(
    {RISK_LEVEL_LOW, RISK_LEVEL_MEDIUM, RISK_LEVEL_HIGH}
)

# -------- 默认超时 / TTL / 轮询参数（PR 3/4 使用） --------

# 工具执行默认超时（秒）。具体工具可在 ToolMetadata 里覆盖。
DEFAULT_TOOL_TIMEOUT_SECONDS: Final[float] = 30.0

# Confirmation token 有效期（秒），默认 10 分钟，涵盖用户"思考 + 确认"。
CONFIRMATION_TOKEN_TTL_SECONDS: Final[int] = 600

# Idempotency 抢占失败后查询已有 record 的轮询参数（PR 3）。
IDEMPOTENCY_POLL_INTERVAL_MS: Final[int] = 200
IDEMPOTENCY_POLL_MAX_ATTEMPTS: Final[int] = 5

# -------- 错误码 --------

# tool_agent 收到"未在 registry 登记"的工具名：启动校验 + 运行时兜底。
ERR_TOOL_NOT_REGISTERED: Final[str] = "tool_not_registered"

# Confirmation token 相关（PR 4 会完整用上，先占位以便各模块共享）。
ERR_TOKEN_INVALID: Final[str] = "confirmation_token_invalid"
ERR_TOKEN_EXPIRED: Final[str] = "confirmation_token_expired"
ERR_TOKEN_MISMATCH: Final[str] = "confirmation_token_mismatch"
# Token 合法，但 token.tool_name 在当前上下文不可见（匿名 × side_effect
# 被 filter_tools_for_auth 拦截，或工具已下架）。必须 fail-closed，而不是
# 悄悄降级到 LLM 路径（否则就等于用 token 绕过了权限校验）。
ERR_TOKEN_AUTH_FORBIDDEN: Final[str] = "confirmation_token_auth_forbidden"

# 超时未知状态，下游请求只能由人工 reconcile。
ERR_TIMEOUT_UNKNOWN: Final[str] = "timeout_unknown"

# Confirmation 签名密钥的环境变量名。
CONFIRMATION_SECRET_ENV: Final[str] = "CONFIRMATION_SECRET"

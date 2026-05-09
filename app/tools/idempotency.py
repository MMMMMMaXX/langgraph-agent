"""工具调用幂等性：key 生成 + 抢占式写入 + 终态查询。

本模块实现设计文档 §4 的"并发抢占式写入"模型：

    INSERT (pending)  —成功—▶ 独占执行权
                      —冲突—▶ 读已有 record，按 status 分流

关键点：
- `compute_idempotency_key` 要**稳定**，同一业务意图两次调用得到同一 key。
  args 做 sorted-keys JSON 归一化，避免字典顺序造成的抖动。
- `acquire_or_resolve` 封装"INSERT-first，冲突回查 + 短轮询 pending"，
  给上层一个干净的三态返回：`winner` / `existing` / `still_pending`。
- `finalize_*` 是状态机出口，统一 status + result_json 的写法，避免
  调用方自己拼字符串。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from app.constants.tool_safety import (
    IDEMPOTENCY_POLL_INTERVAL_MS,
    IDEMPOTENCY_POLL_MAX_ATTEMPTS,
    TOOL_STATUS_FAILED,
    TOOL_STATUS_PENDING,
    TOOL_STATUS_SUCCEEDED,
    TOOL_STATUS_TIMEOUT_UNKNOWN,
)
from app.tools.execution_record import (
    ExecutionRecord,
    ExecutionRecordAlreadyExists,
    get_by_key,
    insert_pending,
    update_status,
)

# ---------- key generation ----------

_KEY_SEPARATOR = "\x1f"  # ASCII Unit Separator — 不会出现在正常业务字符串里


def _normalize_args(args: dict[str, Any]) -> str:
    """归一化 args 为字符串。

    sort_keys=True 保证 dict 顺序变化不影响 key；ensure_ascii=False 保留
    中文字面量，避免 hash 结果因 encoding 差异飘移。
    """

    return json.dumps(args, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_idempotency_key(
    *,
    tenant_id: str,
    user_id: str,
    session_id: str,
    tool_name: str,
    args: dict[str, Any],
) -> str:
    """基于业务维度计算稳定的 idempotency_key。

    同一 (tenant, user, session, tool, 归一化 args) 重复命中同一 key，
    这是 winner-takes-all 语义的前提。
    """

    if not tenant_id or not user_id or not session_id or not tool_name:
        raise ValueError("tenant/user/session/tool must be non-empty")

    normalized_args = _normalize_args(args)
    payload = _KEY_SEPARATOR.join(
        [tenant_id, user_id, session_id, tool_name, normalized_args]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def compute_args_hash(args: dict[str, Any]) -> str:
    """仅对 args 求 hash，供 confirmation token 做 args 不可篡改校验（PR 4）。"""

    digest = hashlib.sha256(_normalize_args(args).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ---------- acquire ----------


class AcquireOutcome(str, Enum):
    """`acquire_or_resolve` 的三态返回。"""

    # 本调用抢占成功，调用方要执行工具并调 finalize_*。
    WINNER = "winner"
    # key 已存在且已到终态 (succeeded / failed / timeout_unknown)，直接返回 record。
    EXISTING = "existing"
    # key 存在但仍 pending（可能别的进程正在执行）；轮询超时后返回该状态。
    STILL_PENDING = "still_pending"


@dataclass(frozen=True)
class AcquireResult:
    outcome: AcquireOutcome
    record: ExecutionRecord


def _is_terminal(status: str) -> bool:
    return status in {
        TOOL_STATUS_SUCCEEDED,
        TOOL_STATUS_FAILED,
        TOOL_STATUS_TIMEOUT_UNKNOWN,
    }


def acquire_or_resolve(
    *,
    idempotency_key: str,
    tenant_id: str,
    user_id: str,
    session_id: str,
    request_id: str,
    tool_name: str,
    args: dict[str, Any],
    path: Path | None = None,
    poll_interval_ms: int = IDEMPOTENCY_POLL_INTERVAL_MS,
    poll_max_attempts: int = IDEMPOTENCY_POLL_MAX_ATTEMPTS,
    sleep: Any = None,
) -> AcquireResult:
    """抢占或解析现有记录。

    流程：
    1. 试 `INSERT status='pending'`；成功 → WINNER。
    2. 主键冲突 → 读已有 record；
       - 终态 → EXISTING；
       - pending → 轮询 `poll_max_attempts` 次，每次 `poll_interval_ms`；
         期间若变终态返回 EXISTING，否则最终 STILL_PENDING。

    `sleep` 参数只用于测试：真实场景留空用 `time.sleep`。
    """

    args_json = _normalize_args(args)
    try:
        record = insert_pending(
            idempotency_key=idempotency_key,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            tool_name=tool_name,
            args_json=args_json,
            path=path,
        )
        return AcquireResult(outcome=AcquireOutcome.WINNER, record=record)
    except ExecutionRecordAlreadyExists:
        pass

    sleep_fn = sleep if sleep is not None else time.sleep
    interval_s = poll_interval_ms / 1000.0

    # 首次冲突后先立刻读一次：避免还没到第一个 sleep 点就已经终态。
    for attempt in range(poll_max_attempts + 1):
        existing = get_by_key(idempotency_key, path=path)
        if existing is None:
            # record 被删除是异常场景；按"谁都抢不到"返回 pending 假状态。
            # 实际生产不预期发生，交给上层决定是否告警。
            raise RuntimeError(
                f"idempotency record vanished after conflict: {idempotency_key}"
            )
        if _is_terminal(existing.status):
            return AcquireResult(outcome=AcquireOutcome.EXISTING, record=existing)
        if attempt == poll_max_attempts:
            break
        sleep_fn(interval_s)

    assert existing is not None
    return AcquireResult(outcome=AcquireOutcome.STILL_PENDING, record=existing)


# ---------- finalize ----------


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def finalize_success(
    idempotency_key: str,
    result: Any,
    *,
    path: Path | None = None,
) -> ExecutionRecord:
    """把记录从 pending 迁到 succeeded，并写入结果。"""

    return update_status(
        idempotency_key=idempotency_key,
        status=TOOL_STATUS_SUCCEEDED,
        result_json=_dumps(result),
        error=None,
        path=path,
    )


def finalize_failure(
    idempotency_key: str,
    error: str,
    *,
    path: Path | None = None,
) -> ExecutionRecord:
    """把记录迁到 failed 并记录 error message。"""

    return update_status(
        idempotency_key=idempotency_key,
        status=TOOL_STATUS_FAILED,
        result_json=None,
        error=error,
        path=path,
    )


def finalize_timeout_unknown(
    idempotency_key: str,
    error: str,
    *,
    path: Path | None = None,
) -> ExecutionRecord:
    """本地超时：状态标 `timeout_unknown`，下游是否成功未知，禁自动重试。

    调用方不得把这视为 `failed`——重试策略对两者必须完全不同：
    - failed 可以安全重试；
    - timeout_unknown 需要人工 reconcile 或下游带 idempotency_key 查询。
    """

    return update_status(
        idempotency_key=idempotency_key,
        status=TOOL_STATUS_TIMEOUT_UNKNOWN,
        result_json=None,
        error=error,
        path=path,
    )


__all__ = [
    "AcquireOutcome",
    "AcquireResult",
    "acquire_or_resolve",
    "compute_args_hash",
    "compute_idempotency_key",
    "finalize_failure",
    "finalize_success",
    "finalize_timeout_unknown",
]

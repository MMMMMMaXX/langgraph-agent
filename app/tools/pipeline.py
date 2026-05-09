"""Side-effect 工具执行管线：把一个"业务意图函数"包成安全调用。

职责：在 LLM function-calling 调到 side_effect 工具时，串起
- 匿名拦截（冗余一层，filter_tools_for_auth 已做第一层）
- idempotency_key 计算
- confirmation token 验签（无 token 则返回 need_confirmation + 新签一张）
- acquire_or_resolve（winner-takes-all）
- 有超时保护地执行（winner 分支）
- finalize_success / finalize_failure / finalize_timeout_unknown

关键约束：
- 返回给 chat_with_tools 的永远是**字符串**（function-calling 协议要求），
  但执行过程中的结构化信息（pending_confirmation、execution record）
  通过 `SideEffectContext` 传回 tool_agent。
- 异常不冒泡——所有失败都落进 record + 返回带前缀的 string，tool_agent
  再决定要不要交给 LLM 二次整合。
- `run_with_timeout` 默认同步直接调用；Phase 1 所有 mock 工具都是本地写入，
  真正接入线上 HTTP 客户端时再换成 `ThreadPoolExecutor.future.result(timeout)`
  实现，本函数作为注入点留给后续迭代。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.auth.context import AuthContext
from app.constants.auth import ERR_ANONYMOUS_FORBIDDEN_SIDE_EFFECT
from app.constants.tool_safety import TOOL_STATUS_SUCCEEDED
from app.tools.confirmation import (
    ConfirmationSecretMissing,
    ConfirmationTokenError,
    issue_token,
    verify_token,
)
from app.tools.execution_record import ExecutionRecord
from app.tools.idempotency import (
    AcquireOutcome,
    acquire_or_resolve,
    compute_idempotency_key,
    finalize_failure,
    finalize_success,
    finalize_timeout_unknown,
)
from app.tools.metadata import ToolMetadata

# 用户可见的返回文案：tool_agent 读 tool_result 时会落到 answer 字段。
_PROMPT_NEED_CONFIRMATION = (
    "该操作会写入系统，需要您确认后再执行。请在下一次请求中携带 confirmation_token。"
)
_PROMPT_ANON_FORBIDDEN = "匿名上下文禁止执行会产生副作用的工具。"
_PROMPT_TOKEN_PREFIX = "确认 token 无效"
_PROMPT_SECRET_MISSING = (
    "服务端未配置 confirmation secret，side_effect 工具暂不可用。"
)
_PROMPT_STILL_PENDING = "该操作仍在执行中，请稍后重试。"
_PROMPT_TIMEOUT_UNKNOWN = "操作本地超时，下游是否成功未知，请联系人工确认。"


@dataclass
class SideEffectContext:
    """一次请求内，side_effect 工具执行需要的上下文。

    - `auth` / `session_id` / `request_id`：构成 idempotency 的维度；
      request_id 只用于 audit，不参与 key hash，避免破坏幂等语义。
    - `confirmation_token`：二次请求携带的 token；空串表示"首次请求"。
    - `now`：时间源，测试里可注入固定值确保可预测。
    - `executions`：累积本轮所有 side_effect 的 record 快照；tool_agent
      从这里读回写入 state。
    - `pending_confirmation`：当有 side_effect 工具返回 need_confirmation 时
      填充；tool_agent 用它覆盖 answer + 写入 state。
    """

    auth: AuthContext
    session_id: str
    request_id: str
    confirmation_token: str = ""
    now: int | None = None
    executions: list[dict] = field(default_factory=list)
    pending_confirmation: dict | None = None


def _record_to_dict(record: ExecutionRecord) -> dict[str, Any]:
    return {
        "idempotency_key": record.idempotency_key,
        "tool_name": record.tool_name,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        # 只回吐 result/err 原貌；不解析 result_json 避免 tool_agent 误用。
        "result_json": record.result_json,
        "error": record.error,
    }


def run_with_timeout(
    fn: Callable[..., Any],
    args: dict[str, Any],
    *,
    timeout_seconds: float,
) -> Any:
    """带超时的工具调用占位实现。

    Phase 1 的 mock 工具全是本地同步调用，直接运行即可；真要做超时隔离时，
    这里会替换为 `concurrent.futures.ThreadPoolExecutor` + `future.result(timeout)`。
    单测可以 monkeypatch 本函数来模拟 `TimeoutError`。
    """

    del timeout_seconds  # placeholder; mock 工具不会超时
    return fn(**args)


def _format_result_for_llm(raw: Any) -> str:
    """把工具返回值转成 LLM 可读的字符串。

    字符串原样返回；dict 走 `repr` 保留 key 顺序但避免依赖 json；其他类型
    统一 str()。function-calling 协议要求 tool message 的 content 是字符串。
    """

    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return repr(raw)
    return str(raw)


def wrap_side_effect_tool(
    name: str,
    fn: Callable[..., Any],
    meta: ToolMetadata,
    ctx: SideEffectContext,
) -> Callable[..., str]:
    """把 `fn(**user_args)` 包成经过完整 pipeline 的 LLM 可调函数。

    内部永远返回字符串：chat_with_tools 会把它回填到 tool message。
    """

    def wrapped(**arguments: Any) -> str:
        # pipeline 可能会额外注入 idempotency_key / tenant_id / user_id 给
        # 下游工具；这里先隔离出 LLM 原始 args，用于 idempotency key 计算，
        # 避免注入字段改变 key hash。
        user_args: dict[str, Any] = {
            k: v
            for k, v in arguments.items()
            if k not in {"idempotency_key", "tenant_id", "user_id"}
        }

        # 1. 匿名二次拦截（防御性）：metadata filter 已移除过 LLM 列表，
        # 走到这里说明上游有 bug，直接拒绝并记一笔 audit。
        if ctx.auth.anonymous:
            ctx.executions.append(
                {
                    "tool_name": name,
                    "status": "rejected_anonymous",
                    "error": ERR_ANONYMOUS_FORBIDDEN_SIDE_EFFECT,
                }
            )
            return _PROMPT_ANON_FORBIDDEN

        # 2. 生成 idempotency_key（所有后续动作都围绕它）。
        idempotency_key = compute_idempotency_key(
            tenant_id=ctx.auth.tenant_id,
            user_id=ctx.auth.user_id,
            session_id=ctx.session_id,
            tool_name=name,
            args=user_args,
        )

        # 3. 没 token：签发一张，返回 need_confirmation。
        if not ctx.confirmation_token:
            try:
                token, payload = issue_token(
                    idempotency_key=idempotency_key,
                    tool_name=name,
                    tenant_id=ctx.auth.tenant_id,
                    user_id=ctx.auth.user_id,
                    args=user_args,
                    now=ctx.now,
                )
            except ConfirmationSecretMissing:
                return _PROMPT_SECRET_MISSING
            ctx.pending_confirmation = {
                "tool_name": name,
                "args": user_args,
                "idempotency_key": idempotency_key,
                "expires_at": payload.expires_at,
                "token": token,
            }
            return _PROMPT_NEED_CONFIRMATION

        # 4. 有 token：校验。任何校验失败都不执行，直接回吐原因。
        try:
            verify_token(
                ctx.confirmation_token,
                tool_name=name,
                tenant_id=ctx.auth.tenant_id,
                user_id=ctx.auth.user_id,
                args=user_args,
                expected_idempotency_key=idempotency_key,
                now=ctx.now,
            )
        except ConfirmationSecretMissing:
            return _PROMPT_SECRET_MISSING
        except ConfirmationTokenError as exc:
            return f"{_PROMPT_TOKEN_PREFIX}: {exc.code}"

        # 5. 抢占式写入 tool_executions。
        acquire = acquire_or_resolve(
            idempotency_key=idempotency_key,
            tenant_id=ctx.auth.tenant_id,
            user_id=ctx.auth.user_id,
            session_id=ctx.session_id,
            request_id=ctx.request_id,
            tool_name=name,
            args=user_args,
        )

        if acquire.outcome == AcquireOutcome.EXISTING:
            ctx.executions.append(_record_to_dict(acquire.record))
            if acquire.record.status == TOOL_STATUS_SUCCEEDED:
                return _format_result_for_llm(acquire.record.result)
            if acquire.record.error:
                return f"该操作此前已失败：{acquire.record.error}"
            return _PROMPT_TIMEOUT_UNKNOWN

        if acquire.outcome == AcquireOutcome.STILL_PENDING:
            ctx.executions.append(_record_to_dict(acquire.record))
            return _PROMPT_STILL_PENDING

        # 6. WINNER：执行工具本体（仍需捕获 timeout / 异常并迁终态）。
        # 注入隐藏字段：对 ticket_create 这种需要知道"当前 tenant/user/key"
        # 的工具是必要的；不关心的工具会通过 **kwargs 吃掉忽略。
        enriched_args = {
            **user_args,
            "idempotency_key": idempotency_key,
            "tenant_id": ctx.auth.tenant_id,
            "user_id": ctx.auth.user_id,
        }
        try:
            raw_result = run_with_timeout(
                fn,
                enriched_args,
                timeout_seconds=meta.timeout_seconds,
            )
        except TimeoutError as exc:
            record = finalize_timeout_unknown(idempotency_key, str(exc) or "timeout")
            ctx.executions.append(_record_to_dict(record))
            return _PROMPT_TIMEOUT_UNKNOWN
        except Exception as exc:  # noqa: BLE001 — tool 错误不冒泡，统一落 failed
            record = finalize_failure(idempotency_key, f"{type(exc).__name__}: {exc}")
            ctx.executions.append(_record_to_dict(record))
            return f"工具执行失败：{exc}"

        record = finalize_success(idempotency_key, raw_result)
        ctx.executions.append(_record_to_dict(record))
        return _format_result_for_llm(raw_result)

    return wrapped


def prepare_side_effect_impls(
    tools: list[dict[str, Any]],
    tool_impls: dict[str, Callable[..., Any]],
    metadata_lookup: Callable[[str], ToolMetadata],
    ctx: SideEffectContext,
) -> dict[str, Callable[..., Any]]:
    """对传入 tools 列表里所有 side_effect 工具做 pipeline 包装。

    read_only 工具保持原样，避免不必要的代理开销，也保留现有测试语义。
    """

    result: dict[str, Callable[..., Any]] = {}
    for spec in tools:
        name = spec["function"]["name"]
        if name not in tool_impls:
            continue
        meta = metadata_lookup(name)
        if meta.side_effect:
            result[name] = wrap_side_effect_tool(
                name, tool_impls[name], meta, ctx
            )
        else:
            result[name] = tool_impls[name]
    return result


__all__ = [
    "SideEffectContext",
    "prepare_side_effect_impls",
    "run_with_timeout",
    "wrap_side_effect_tool",
]

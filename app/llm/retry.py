"""LLM/Embedding 调用的重试、超时、错误包装策略。

这一层与具体 provider/profile 无关，只做：
- 环境变量驱动的 timeout / retry 参数读取
- 通用的 double-retry + 指数退避
- 统一异常封装 `LLMCallError`
"""

import time

from app.llm._helpers import _env_float, _env_int
from app.utils.errors import classify_exception, format_exception_message


# --- Retry / timeout 默认值 -------------------------------------------------
# 这些默认值经过简单压测选取，生产场景建议通过环境变量覆盖。
DEFAULT_CHAT_TIMEOUT_SECONDS = 45.0
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 30.0
DEFAULT_LLM_MAX_RETRIES = 2
DEFAULT_LLM_RETRY_BACKOFF_SECONDS = 0.5

# 某些错误重试无意义（鉴权/参数错误），快速失败即可。
NON_RETRYABLE_ERROR_CODES = {"auth_error", "bad_request"}


class LLMCallError(RuntimeError):
    """统一封装 LLM SDK 调用失败。

    目的不是隐藏原始异常，而是把上游可能抛出的各种网络/限流/参数异常，
    收敛成项目内部统一可识别的一种错误类型，方便 agent 层做降级处理。
    """

    def __init__(
        self,
        code: str,
        message: str,
        profile: str = "",
        provider: str = "",
        model: str = "",
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.profile = profile
        self.provider = provider
        self.model = model


def _get_request_timeout(kind: str) -> float:
    if kind == "embedding":
        return _env_float(
            "EMBEDDING_TIMEOUT_SECONDS", DEFAULT_EMBEDDING_TIMEOUT_SECONDS
        )
    return _env_float("LLM_TIMEOUT_SECONDS", DEFAULT_CHAT_TIMEOUT_SECONDS)


def _get_max_retries(kind: str) -> int:
    if kind == "embedding":
        return max(0, _env_int("EMBEDDING_MAX_RETRIES", DEFAULT_LLM_MAX_RETRIES))
    return max(0, _env_int("LLM_MAX_RETRIES", DEFAULT_LLM_MAX_RETRIES))


def _get_retry_backoff_seconds(kind: str) -> float:
    if kind == "embedding":
        return max(
            0.0,
            _env_float(
                "EMBEDDING_RETRY_BACKOFF_SECONDS",
                DEFAULT_LLM_RETRY_BACKOFF_SECONDS,
            ),
        )
    return max(
        0.0,
        _env_float("LLM_RETRY_BACKOFF_SECONDS", DEFAULT_LLM_RETRY_BACKOFF_SECONDS),
    )


def _should_retry_exception(exc: Exception) -> bool:
    code = classify_exception(exc)
    return code not in NON_RETRYABLE_ERROR_CODES


def _raise_llm_call_error(
    exc: Exception,
    *,
    profile: str,
    provider: str,
    model: str,
) -> None:
    raise LLMCallError(
        code=classify_exception(exc),
        message=format_exception_message(exc),
        profile=profile,
        provider=provider,
        model=model,
    ) from exc


def _call_with_retry(
    operation,
    *,
    kind: str,
    profile: str,
    provider: str,
    model: str,
):
    """执行一次模型请求，带显式 timeout 和有限重试。

    SDK 自身重试关闭后，所有 chat/embedding 调用都走这里，避免网络抖动
    或限流时把整个 graph 永久挂住。
    """

    max_retries = _get_max_retries(kind)
    backoff_seconds = _get_retry_backoff_seconds(kind)
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _should_retry_exception(exc):
                break
            time.sleep(backoff_seconds * (2**attempt))

    _raise_llm_call_error(
        last_exc or RuntimeError("unknown llm call failure"),
        profile=profile,
        provider=provider,
        model=model,
    )

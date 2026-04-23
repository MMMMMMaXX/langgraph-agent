def format_exception_message(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def classify_exception(exc: Exception, preferred_code: str | None = None) -> str:
    """把零散异常收敛成项目内统一错误码。"""
    if preferred_code:
        return preferred_code

    raw = f"{exc.__class__.__name__} {exc}".lower()

    if any(token in raw for token in ["timeout", "timed out", "readtimeout"]):
        return "timeout"
    if any(token in raw for token in ["rate limit", "ratelimit", "429"]):
        return "rate_limit"
    if any(token in raw for token in ["401", "403", "auth", "unauthorized", "forbidden"]):
        return "auth_error"
    if any(
        token in raw
        for token in ["400", "badrequest", "invalid", "parameter", "argument"]
    ):
        return "bad_request"
    if any(
        token in raw
        for token in ["connection", "connect", "dns", "network", "unreachable"]
    ):
        return "network_error"
    if any(token in raw for token in ["json", "decode", "parse", "schema"]):
        return "bad_response"

    return "internal_error"


def build_error_info(
    exc: Exception,
    *,
    stage: str,
    source: str,
    preferred_code: str | None = None,
) -> dict:
    """构造统一错误结构，供 debug / log / eval 使用。"""
    code = getattr(exc, "code", None) or classify_exception(
        exc, preferred_code=preferred_code
    )
    return {
        "code": code,
        "stage": stage,
        "source": source,
        "message": format_exception_message(exc),
    }

"""统一的 prompt / chunk / token 脱敏入口（PR-0）。

设计目标（监控方案 §5、§6、§12）：
- log、trace、metrics label 全部经过同一份 redactor，避免双份逻辑；
- 命中疑似敏感串时替换为 `REDACTED_PLACEHOLDER`，可被检索定位；
- 提供 `preview()` 仅返回截断 + 脱敏后的预览，禁止任何路径返回原文。

本模块只做规则匹配，不做业务策略。具体哪些字段允许进 sink 由调用方决定。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from app.constants.observability import (
    REDACTED_PLACEHOLDER,
    TRACE_PREVIEW_MAX_CHARS,
)

# ---- 敏感模式 -----------------------------------------------------------
#
# 这里只列出**项目中已知会出现**的 token 形态，避免过度匹配把正常文本误伤。
# 新增形态需配套单测，不允许私下扩展。

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # OpenAI / Anthropic / 通用 sk- 前缀 key
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    # Bearer / Authorization header value
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.=]{16,}\b"),
    # Confirmation token：本项目 tool confirmation 使用 `confirm-` / `cnf_` 前缀
    re.compile(r"\bconfirm[-_][A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\bcnf_[A-Za-z0-9]{8,}\b"),
    # 通用 JWT（三段 base64url，长度足够区分于普通短串）
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),
    # 注：曾经有一条 `[a-fA-F0-9]{32,}` 通用 hex 规则，但会把 40 位 git_sha
    # 误判为 API key、把 app_version_info 的 git_sha label 整个删掉。
    # 决定移除——上面四条专属前缀已经覆盖项目里实际出现过的密钥形态；
    # 真要捕获裸 hex key，需要在调用点按 label 名（如 api_key/token）显式拦。
)

# 不允许进 metrics label 的字段名（小写比较）。日志 / trace 仍可保留 preview，
# 但要先过 `redact_text`。
_BANNED_LABEL_FIELDS: frozenset[str] = frozenset(
    {
        "query",
        "prompt",
        "answer",
        "content",
        "chunk",
        "user_id",
        "session_id",
        "doc_id",
        "chunk_id",
        "request_id",
        "authorization",
        "api_key",
        "token",
        "confirmation_token",
    }
)


def redact_text(text: str) -> str:
    """对一段文本执行 token-shaped 串脱敏。

    匹配命中即整段替换为 `REDACTED_PLACEHOLDER`；未命中则原样返回。
    本函数**不做截断**，调用方按需配合 `preview()`。
    """

    if not text:
        return text
    redacted = text
    for pattern in _PATTERNS:
        redacted = pattern.sub(REDACTED_PLACEHOLDER, redacted)
    return redacted


def preview(text: str, limit: int = TRACE_PREVIEW_MAX_CHARS) -> str:
    """返回截断 + 脱敏后的预览串。永远不返回原文。"""

    if text is None:
        return ""
    redacted = redact_text(str(text))
    if len(redacted) <= limit:
        return redacted
    return redacted[:limit] + "..."


def is_banned_label_field(field_name: str) -> bool:
    """判断字段名是否禁止作为 metrics label。"""

    return field_name.lower() in _BANNED_LABEL_FIELDS


def looks_sensitive(value: str) -> bool:
    """判断单个值是否疑似敏感串，便于 emit 包装器对 label 取值兜底。"""

    if not value:
        return False
    return any(p.search(value) for p in _PATTERNS)


def redact_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """对 dict-like 结构递归脱敏。值为 str 走 `redact_text`，其它原样返回。"""

    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            result[key] = redact_text(value)
        elif isinstance(value, Mapping):
            result[key] = redact_mapping(value)
        elif isinstance(value, list):
            result[key] = [
                (
                    redact_text(item)
                    if isinstance(item, str)
                    else (redact_mapping(item) if isinstance(item, Mapping) else item)
                )
                for item in value
            ]
        else:
            result[key] = value
    return result


__all__ = [
    "is_banned_label_field",
    "looks_sensitive",
    "preview",
    "redact_mapping",
    "redact_text",
]

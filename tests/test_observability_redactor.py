"""PR-0：redactor 单测。

覆盖监控方案 PR-0 验收：prompt、chunk、API key、confirmation token 四类样本。
"""

from __future__ import annotations

import pytest

from app.constants.observability import REDACTED_PLACEHOLDER
from app.observability.redactor import (
    is_banned_label_field,
    looks_sensitive,
    preview,
    redact_mapping,
    redact_text,
)


def test_redact_text_handles_openai_style_api_key() -> None:
    text = "OpenAI key: sk-abcdef0123456789ABCDEF0123 in config"
    out = redact_text(text)
    assert "sk-" not in out
    assert REDACTED_PLACEHOLDER in out


def test_redact_text_handles_bearer_token_in_header() -> None:
    text = "Authorization: Bearer eyJabcDEF1234567890.aaaaaaaa.bbbbbbbb"
    out = redact_text(text)
    assert REDACTED_PLACEHOLDER in out
    # JWT 也应被独立模式拦下；Bearer 也应被拦下，最终原 token 不残留
    assert "eyJabcDEF" not in out
    assert "Bearer e" not in out


def test_redact_text_handles_confirmation_token_variants() -> None:
    # 项目内两种风格：confirm-XXXX 与 cnf_XXXX
    cases = [
        "需要确认：confirm-AbCdEf12345 才能执行",
        "ticket cnf_9zZqAa01 已生成",
    ]
    for text in cases:
        assert REDACTED_PLACEHOLDER in redact_text(text)


def test_redact_text_keeps_normal_prompt_text() -> None:
    """普通中文 prompt / chunk 不应被误伤。"""

    text = "请总结这段文档：王者荣耀新版本上线了三位新英雄。"
    assert redact_text(text) == text


def test_preview_truncates_and_redacts_chunk() -> None:
    chunk = "正文片段开头 sk-abcdef0123456789ABCDEF0123 后面还有很多内容" + "x" * 200
    out = preview(chunk, limit=80)
    assert "sk-" not in out
    assert out.endswith("...")
    assert len(out) <= 80 + len("...")


def test_redact_mapping_recurses_dict_and_list() -> None:
    payload = {
        "outer": "Bearer eyJabcDEF1234567890.aaaaaaaa.bbbbbbbb",
        "nested": {"api_key": "sk-abcdef0123456789ABCDEF0123"},
        "items": ["normal", "confirm-AbCdEf12345"],
    }
    out = redact_mapping(payload)
    assert REDACTED_PLACEHOLDER in out["outer"]
    assert REDACTED_PLACEHOLDER in out["nested"]["api_key"]
    assert out["items"][0] == "normal"
    assert REDACTED_PLACEHOLDER in out["items"][1]


def test_is_banned_label_field_covers_high_cardinality_keys() -> None:
    for key in ("user_id", "query", "prompt", "doc_id", "chunk_id"):
        assert is_banned_label_field(key)
    for key in ("route_template", "provider", "model", "tool_name"):
        assert not is_banned_label_field(key)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("sk-abcdef0123456789ABCDEF0123", True),
        ("confirm-AbCdEf12345", True),
        ("普通文本", False),
        ("", False),
        # git_sha（40 位 hex）属于合法元数据，不应被通用 hex 规则误判为敏感串。
        ("a3f5b6c7d8e9012345678901234567890abcdef0", False),
        # 短 hex 串同样不该误伤
        ("deadbeef", False),
    ],
)
def test_looks_sensitive(value: str, expected: bool) -> None:
    assert looks_sensitive(value) is expected


def test_redact_text_does_not_clobber_git_sha() -> None:
    """旧版 32+ hex 规则会把 git_sha 整段擦掉，导致 app_version_info label 丢失。

    本用例是回归保护：保证 redact_text 对纯 hex 串保留原文。
    """

    git_sha = "a3f5b6c7d8e9012345678901234567890abcdef0"
    text = f"version=v1.2.3 git_sha={git_sha}"
    assert redact_text(text) == text

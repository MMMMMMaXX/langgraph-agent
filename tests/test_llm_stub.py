"""验证 llm_stub fixture 本身工作正常。

作为后续业务测试的样板：
- **不要** `from app.llm import chat`，那样会在 import 时抓住原函数，fixture 替换不生效
- 在测试里用 `app.llm.chat(...)` 现查现用，或在 fixture 后再 import
- 业务代码只要在**函数体内部**调用 `chat(...)`，monkeypatch 会命中
"""

from __future__ import annotations

import app.llm as llm_pkg


def test_llm_stub_returns_default(llm_stub) -> None:  # type: ignore[no-untyped-def]
    result = llm_pkg.chat([{"role": "user", "content": "ping"}])
    assert result == "stub-answer"
    assert len(llm_stub.calls) == 1
    assert llm_stub.calls[0]["messages"][0]["content"] == "ping"


def test_llm_stub_custom_response(llm_stub) -> None:  # type: ignore[no-untyped-def]
    llm_stub.set_response("hello world")
    assert llm_pkg.chat([{"role": "user", "content": "x"}]) == "hello world"


def test_llm_stub_response_fn(llm_stub) -> None:  # type: ignore[no-untyped-def]
    llm_stub.set_response_fn(lambda messages, **_: f"echo:{messages[-1]['content']}")
    assert llm_pkg.chat([{"role": "user", "content": "abc"}]) == "echo:abc"

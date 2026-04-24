"""共享 pytest fixture。

这里放**所有测试都能用**的 fixture，尤其是 LLM mock —— 确保：
1. CI 里跑测试不需要真实 API key
2. 单测速度可控、结果稳定
3. 不会不小心把测试请求打到真实 provider 花钱 / 触发风控
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest


# LLM mock 配置：默认返回固定字符串；测试可覆盖 set_response() 改响应
class _LLMStub:
    """monkey-patch 替换 app.llm.chat.chat 的桩。

    - calls：记录所有调用参数，断言 LLM 被正确调用
    - set_response(str) / set_response_fn(callable)：自定义返回
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fn: Callable[..., str] = lambda **_: "stub-answer"

    def __call__(
        self,
        messages: list[dict],
        max_completion_tokens: int | None = None,
        on_delta: Callable[[str], None] | None = None,
        profile: str = "default_chat",
        trace_stage: str = "",
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "profile": profile,
                "trace_stage": trace_stage,
                "max_completion_tokens": max_completion_tokens,
            }
        )
        result = self._fn(
            messages=messages,
            profile=profile,
            trace_stage=trace_stage,
        )
        if on_delta is not None:
            on_delta(result)
        return result

    def set_response(self, text: str) -> None:
        self._fn = lambda **_: text

    def set_response_fn(self, fn: Callable[..., str]) -> None:
        self._fn = fn


@pytest.fixture
def llm_stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[_LLMStub]:
    """替换项目里 LLM chat 入口。

    用法：
        def test_foo(llm_stub):
            llm_stub.set_response("hello")
            ...  # 被测代码内部调用 chat(...) 会得到 "hello"
            assert len(llm_stub.calls) == 1
    """

    stub = _LLMStub()
    # 业务代码多处通过 `from app.llm import chat` 再调用 chat(...)，
    # 因此必须 patch app.llm 包下的绑定，同时 patch app.llm.chat 模块下的定义。
    # 注意：`app.llm.chat` 同名函数在 app/llm/__init__.py 被 re-export 后，
    # 包属性 `app.llm.chat` 会指向函数而非子模块，所以需要从 sys.modules 拿子模块。
    import sys

    import app.llm as llm_pkg

    llm_chat_mod = sys.modules["app.llm.chat"]

    monkeypatch.setattr(llm_chat_mod, "chat", stub)
    monkeypatch.setattr(llm_pkg, "chat", stub)
    yield stub

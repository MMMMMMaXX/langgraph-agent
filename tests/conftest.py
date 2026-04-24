"""共享 pytest fixture。

这里放**所有测试都能用**的 fixture，尤其是 LLM mock —— 确保：
1. CI 里跑测试不需要真实 API key
2. 单测速度可控、结果稳定
3. 不会不小心把测试请求打到真实 provider 花钱 / 触发风控
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# 必须在 import app.* 之前设置：
# app.llm.providers 在模块加载时就会把 PROVIDER_CONFIGS 冻住（读 env 的快照）。
# 之后再用 monkeypatch.setenv 不会影响已经构建好的 registry，readiness 检查就会
# 报 "model empty" / "API_KEY not set"。
# 这里给出 CI 里也一定成立的最小可用环境，本地有 .env 不受影响（setdefault 不覆盖）。
# ---------------------------------------------------------------------------
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-chat")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

from collections.abc import Callable, Iterator  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402


def _fake_sdk_response(
    content: str = "",
    tool_calls: list[Any] | None = None,
) -> Any:
    """造一个最小的 SDK-shape 返回。

    - 默认只填 `choices[0].message.content`（普通问答 / plan_routes 路径够用）
    - `tool_calls` 不为空时，还给 message 挂上 tool_calls 字段 + model_dump()，
      供 `chat_with_tools` 里 `followup_messages.append(first_message.model_dump())`
      使用。
    """
    msg_dict: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg_dict["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    message.model_dump = lambda: msg_dict
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_tool_call(
    name: str, arguments: dict[str, Any], call_id: str = "call_1"
) -> Any:
    """测试辅助：造一个 SDK function-call 风格的 tool_call 对象。

    arguments 会被 json.dumps 成字符串（OpenAI SDK 原生就是这样返回的）。
    """
    import json

    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


# LLM mock 配置：默认返回固定字符串；测试可覆盖 set_response() 改响应
class _LLMStub:
    """monkey-patch 替换 app.llm.chat.chat 的桩。

    - calls：记录所有调用参数，断言 LLM 被正确调用
    - set_response(str) / set_response_fn(callable)：自定义返回

    同时拦截 `_create_chat_completion`：项目内像 `plan_routes` / `rewrite_query`
    / `chat_with_tools` 这类"薄包装"并不走 `chat()`，而是直接调底层
    `_create_chat_completion` 拿 SDK 原生响应对象。为了让这些链路在测试里也能被
    llm_stub 接管，这里伪造一个最小 SDK-shape 响应（choices[0].message.content），
    内容仍由 set_response* 决定。

    set_response_fn 的返回值既可以是纯 str（用作 content），也可以是 dict
    `{"content": str, "tool_calls": [...]}`：后者专门用于 tool_agent 的
    tool_select 阶段，测试通过 trace_stage 区分不同阶段给出不同响应。
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
                "via": "chat",
            }
        )
        result = self._fn(
            messages=messages,
            profile=profile,
            trace_stage=trace_stage,
        )
        # 兼容 tool_agent 场景：set_response_fn 可能按 trace_stage 分派返回 dict
        if isinstance(result, dict):
            result = result.get("content", "")
        if on_delta is not None:
            on_delta(result)
        return result

    def set_response(self, text: str) -> None:
        self._fn = lambda **_: text

    def set_response_fn(self, fn: Callable[..., str]) -> None:
        self._fn = fn

    def create_completion(
        self,
        profile: str = "default_chat",
        trace_stage: str = "",
        **kwargs: Any,
    ) -> Any:
        """模拟 `_create_chat_completion` —— 返回 SDK-shape 对象。

        `messages` 从 kwargs 取，便于记录；其余 kwargs（max_completion_tokens 等）
        也会被原封不动地记下，方便断言。
        """
        messages = kwargs.get("messages") or []
        self.calls.append(
            {
                "messages": messages,
                "profile": profile,
                "trace_stage": trace_stage,
                "max_completion_tokens": kwargs.get("max_completion_tokens"),
                "via": "_create_chat_completion",
            }
        )
        content = self._fn(
            messages=messages,
            profile=profile,
            trace_stage=trace_stage,
        )
        # dict 返回：tool_calls 路径；str 返回：普通文本
        if isinstance(content, dict):
            return _fake_sdk_response(
                content=content.get("content", ""),
                tool_calls=content.get("tool_calls"),
            )
        return _fake_sdk_response(content)


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
    original_chat = llm_chat_mod.chat

    monkeypatch.setattr(llm_chat_mod, "chat", stub)
    monkeypatch.setattr(llm_pkg, "chat", stub)
    # 扫所有已加载模块：凡是通过 `from app.llm import chat` 绑到本模块 namespace
    # 的 chat 引用（== 原函数对象），统一替换为 stub。否则业务模块在 import 时
    # 就抓住了原函数，后续 patch 只改包属性对它们无效。
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod_name.startswith("app.llm"):
            continue
        if getattr(mod, "chat", None) is original_chat:
            monkeypatch.setattr(mod, "chat", stub)
    # 拦截底层入口：plan_routes / rewrite_query 这类直接用 SDK 响应的函数会走这里
    monkeypatch.setattr(llm_chat_mod, "_create_chat_completion", stub.create_completion)
    yield stub

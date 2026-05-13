from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

StreamCallback = Callable[[str, dict[str, Any]], None]


# stream_callback 是请求运行态对象，不属于可持久化 graph state。
# 使用 ContextVar 可以让同一进程内并发请求各自拿到自己的 SSE 回调，
# 同时避免 callable 被官方 SqliteSaver 写入 checkpoint。
_stream_callback: ContextVar[StreamCallback | None] = ContextVar(
    "stream_callback",
    default=None,
)


# Confirmation token 原文同样禁止进入 AgentState：LangGraph 会把节点返回的 state
# 拍进 checkpoint，并随 run 输出被 LangSmith 记录。token 一旦落到这两层，就等于
# 绕过了"只在 API 响应短暂出现"的安全边界。
#
# 实现上用一个 "mutable list 容器 + ContextVar" 而不是直接 set 一个 str。
# 原因：LangGraph 用 `contextvars.Context.run(...)` 隔离节点执行上下文，**节点里
# 的 ContextVar 写入不会传回父上下文**。但父上下文放进去的可变对象（list）
# 会被子上下文共享引用，子节点 `append` 的结果在父上下文可见。
# 因此：chat_service 在请求开始时把一个空 list 塞进 ContextVar，节点 append
# token，出口端读 list 最后一项并重置。
_pending_confirmation_token_holder: ContextVar[list[str] | None] = ContextVar(
    "pending_confirmation_token_holder",
    default=None,
)


def set_stream_callback(callback: StreamCallback | None):
    """设置当前请求的流式回调，返回 token 供 finally 中恢复上下文。"""

    return _stream_callback.set(callback)


def reset_stream_callback(token) -> None:
    """恢复进入请求前的流式回调上下文。"""

    _stream_callback.reset(token)


def get_stream_callback() -> StreamCallback | None:
    """读取当前请求的流式回调。"""

    return _stream_callback.get()


def begin_pending_confirmation_scope():
    """请求入口：绑定一个新的 holder list，返回 ctx token 供 finally reset。

    只有在 holder 不为 None 时，节点里的 `set_pending_confirmation_token` 才生效。
    这样单测/裸调节点也不会误把 token 写到不存在的 holder 上。
    """

    return _pending_confirmation_token_holder.set([])


def reset_pending_confirmation_token(ctx_token) -> None:
    """请求生命周期结束时恢复上下文，确保 token holder 不跨请求残留。"""

    _pending_confirmation_token_holder.reset(ctx_token)


def set_pending_confirmation_token(token: str) -> None:
    """节点侧：把 pipeline 生成的原文 token 压入当前请求的 holder。

    LangGraph 节点跑在子 Context 里，直接 `ContextVar.set(str)` 无法被父上下文
    看到；但父上下文塞进 ContextVar 的 list 对象是共享引用，`append` 能跨上下
    文边界生效。
    """

    holder = _pending_confirmation_token_holder.get()
    if holder is None or not token:
        return
    holder.append(token)


def get_pending_confirmation_token() -> str:
    """读取当前请求的 confirmation token 原文（可能为空字符串）。

    取 holder 里最后一条，避免多节点先后 push 时前者覆盖。当前 graph 每轮最多
    一次 need_confirmation，所以无论首尾都一样；保留 last 语义更稳妥。
    """

    holder = _pending_confirmation_token_holder.get()
    if not holder:
        return ""
    return holder[-1]

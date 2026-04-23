"""Chat completion 相关接口。

包含：
- `_create_chat_completion`：底层 chat completion 封装（解析 provider/profile + 超时重试 + tracing）
- `chat` / `chat_with_tools`：通用对话接口
- 任务专用薄封装：`rewrite_query` / `summarize_messages` / `plan_routes`

任务封装放在这里的原因是它们都只是 chat 的固定 prompt 模板，暂不值得独立模块。
后续如果 routing / summary 逻辑继续膨胀，可以再拆出去。
"""

import json
from collections.abc import Callable

from app.constants.model_profiles import (
    PROFILE_DEFAULT_CHAT,
    PROFILE_REWRITE,
    PROFILE_ROUTING,
    PROFILE_SUMMARY,
)
from app.constants.routes import (
    ROUTE_CHAT_AGENT,
    ROUTE_NOVEL_SCRIPT_AGENT,
    ROUTE_RAG_AGENT,
    ROUTE_TOOL_AGENT,
)
from app.llm.caching import _get_client
from app.llm.providers import (
    _resolve_model,
    _resolve_profile,
    _resolve_provider,
)
from app.llm.retry import (
    LLMCallError,
    _call_with_retry,
    _get_max_retries,
    _get_request_timeout,
)
from app.prompts.chat import (
    SUMMARIZE_MESSAGES_SYSTEM_PROMPT,
    build_summarize_messages_user_prompt,
)
from app.prompts.routing import (
    PLAN_ROUTES_SYSTEM_PROMPT,
    REWRITE_QUERY_SYSTEM_PROMPT,
    build_route_planning_user_prompt,
)
from app.tracing import (
    add_current_run_metadata,
    build_model_trace_metadata,
    next_model_call_index,
)
from app.utils.logger import log_warning


def _create_chat_completion(
    profile: str = PROFILE_DEFAULT_CHAT,
    trace_stage: str = "",
    **kwargs,
):
    """统一执行 chat completion 请求，并把底层 SDK 异常转换成项目内标准错误。"""

    profile_config = _resolve_profile(profile, kind="chat")
    provider_config = _resolve_provider(profile_config)
    model = kwargs.pop("model", None) or _resolve_model(profile_config, provider_config)
    client = _get_client(provider_config)
    call_index = next_model_call_index()
    timeout = _get_request_timeout("chat")
    max_retries = _get_max_retries("chat")
    add_current_run_metadata(
        build_model_trace_metadata(
            kind="chat",
            profile=profile_config.name,
            provider=provider_config.name,
            model=model,
            stage=trace_stage or profile_config.name,
            call_index=call_index,
            streaming=bool(kwargs.get("stream", False)),
            input_count=len(kwargs.get("messages") or []),
            timeout_seconds=timeout,
            max_retries=max_retries,
        ),
        event_name="model_call",
    )

    return _call_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            timeout=timeout,
            **kwargs,
        ),
        kind="chat",
        profile=profile_config.name,
        provider=provider_config.name,
        model=model,
    )


def chat(
    messages: list[dict],
    max_completion_tokens: int | None = None,
    on_delta: Callable[[str], None] | None = None,
    profile: str = PROFILE_DEFAULT_CHAT,
    trace_stage: str = "",
) -> str:
    """项目内最通用的纯文本对话接口。

    - 非流式：直接返回完整字符串
    - 流式：边消费 SDK stream，边把 delta 回调给上层
    - profile：决定本次调用默认使用哪个 provider / model

    无论哪种模式，只要底层 SDK 调用失败，都会统一抛出 `LLMCallError`，
    由 agent 层决定是否降级、fallback 或返回用户可读提示。
    """
    request_kwargs = {
        "messages": messages,
    }
    if max_completion_tokens is not None:
        request_kwargs["max_completion_tokens"] = max_completion_tokens

    if on_delta is None:
        res = _create_chat_completion(
            profile=profile,
            trace_stage=trace_stage,
            **request_kwargs,
        )
        return (res.choices[0].message.content or "").strip()

    stream = _create_chat_completion(
        profile=profile,
        trace_stage=trace_stage,
        **request_kwargs,
        stream=True,
    )
    chunks: list[str] = []

    for event in stream:
        if not event.choices:
            continue

        delta = event.choices[0].delta.content or ""
        if not delta:
            continue

        chunks.append(delta)
        on_delta(delta)

    return "".join(chunks).strip()


def chat_with_tools(
    messages: list[dict],
    tools: list[dict],
    tool_impls: dict[str, Callable],
    max_completion_tokens: int | None = None,
    finalize_with_llm: bool = True,
    on_delta: Callable[[str], None] | None = None,
    profile: str = PROFILE_DEFAULT_CHAT,
) -> dict:
    """带 function calling 的统一工具对话接口。"""
    request_kwargs = {
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }
    if max_completion_tokens is not None:
        request_kwargs["max_completion_tokens"] = max_completion_tokens

    first_response = _create_chat_completion(
        profile=profile,
        trace_stage="tool_select",
        **request_kwargs,
    )
    first_message = first_response.choices[0].message
    tool_calls = first_message.tool_calls or []

    if not tool_calls:
        return {
            "answer": (first_message.content or "").strip(),
            "tool_calls": [],
            "tool_results": [],
        }

    followup_messages = list(messages)
    followup_messages.append(first_message.model_dump())
    executed_tool_calls = []
    tool_results = []

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        raw_arguments = tool_call.function.arguments or "{}"

        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            log_warning(
                "chat_with_tools",
                "invalid tool arguments JSON; fallback to empty arguments",
                {
                    "tool_name": tool_name,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "arguments_preview": raw_arguments[:160],
                },
            )
            arguments = {}

        tool_fn = tool_impls.get(tool_name)
        if tool_fn is None:
            tool_output = f"工具 {tool_name} 不存在。"
        else:
            try:
                tool_output = tool_fn(**arguments)
            except TypeError:
                tool_output = f"工具 {tool_name} 参数错误：{arguments}"
            except Exception as exc:
                tool_output = f"工具 {tool_name} 执行失败：{str(exc)}"

        executed_tool_calls.append(
            {
                "id": tool_call.id,
                "name": tool_name,
                "arguments": arguments,
            }
        )
        tool_results.append(
            {
                "id": tool_call.id,
                "name": tool_name,
                "output": tool_output,
            }
        )

        followup_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_output,
            }
        )

    final_answer = ""
    if finalize_with_llm:
        final_answer = chat(
            followup_messages,
            max_completion_tokens=max_completion_tokens,
            on_delta=on_delta,
            profile=profile,
            trace_stage="tool_finalize",
        )

    return {
        "answer": final_answer,
        "tool_calls": executed_tool_calls,
        "tool_results": tool_results,
    }


def rewrite_query(messages: list[dict], profile: str = PROFILE_REWRITE) -> str:
    """把上下文相关追问改写成适合检索的完整问题。"""
    res = _create_chat_completion(
        profile=profile,
        trace_stage="rewrite_query",
        messages=[
            {
                "role": "system",
                "content": REWRITE_QUERY_SYSTEM_PROMPT,
            },
            *messages,
        ],
    )
    return res.choices[0].message.content.strip()


def summarize_messages(
    old_summary: str, messages: list[dict], profile: str = PROFILE_SUMMARY
) -> str:
    """把旧摘要和新增对话压缩成新的会话状态摘要。"""
    latest_user_message = ""
    for m in reversed(messages):
        if m["role"] == "user":
            latest_user_message = m["content"]
            break

    recent_dialogue = "\n".join([f'{m["role"]}: {m["content"]}' for m in messages])

    res = _create_chat_completion(
        profile=profile,
        trace_stage="summarize_messages",
        messages=[
            {
                "role": "system",
                "content": SUMMARIZE_MESSAGES_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_summarize_messages_user_prompt(
                    old_summary=old_summary,
                    latest_user_message=latest_user_message,
                    recent_dialogue=recent_dialogue,
                ),
            },
        ],
    )

    return res.choices[0].message.content.strip()


def plan_routes(message: str, profile: str = PROFILE_ROUTING) -> list[str]:
    """根据用户问题为 supervisor 提供 agent 路由建议。"""
    try:
        res = _create_chat_completion(
            profile=profile,
            trace_stage="plan_routes",
            messages=[
                {
                    "role": "system",
                    "content": PLAN_ROUTES_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_route_planning_user_prompt(message),
                },
            ],
        )
    except LLMCallError as exc:
        log_warning(
            "plan_routes",
            "route planning LLM failed; fallback to chat_agent",
            {
                "code": exc.code,
                "profile": exc.profile,
                "provider": exc.provider,
                "model": exc.model,
            },
        )
        return [ROUTE_CHAT_AGENT]

    text = res.choices[0].message.content.strip()

    try:
        routes = json.loads(text)
        allowed = {
            ROUTE_TOOL_AGENT,
            ROUTE_RAG_AGENT,
            ROUTE_CHAT_AGENT,
            ROUTE_NOVEL_SCRIPT_AGENT,
        }
        routes = [r for r in routes if r in allowed]
        return routes or [ROUTE_CHAT_AGENT]
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        log_warning(
            "plan_routes",
            "invalid route planning response; fallback to chat_agent",
            {
                "error": f"{exc.__class__.__name__}: {exc}",
                "response_preview": text[:160],
            },
        )
        return [ROUTE_CHAT_AGENT]

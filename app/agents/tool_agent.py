from app.constants.model_profiles import PROFILE_TOOL_CHAT
from app.constants.routes import ROUTE_TOOL_AGENT
from app.constants.tool_safety import ERR_TOKEN_AUTH_FORBIDDEN
from app.constants.tooling import (
    TOOL_MULTI_INTENT_KEYWORDS,
    TOOL_NAME_CALCULATE,
    TOOL_NAME_GET_WEATHER,
    TOOL_NAME_MONITOR_QUERY_ERRORS,
    TOOL_NAME_TICKET_CREATE,
    TOOL_TYPE_NONE,
)
from app.llm import chat_with_tools, get_profile_runtime_info
from app.prompts.tooling import TOOL_AGENT_SYSTEM_PROMPT
from app.runtime_context import set_pending_confirmation_token
from app.state import AgentState
from app.streaming import build_answer_streamer
from app.tools.confirmation import (
    ConfirmationSecretMissing,
    ConfirmationTokenError,
    decode_signed_payload,
    redact_pending_confirmation,
)
from app.tools.metadata import filter_tools_for_auth, get_tool_metadata
from app.tools.monitor import monitor_query_errors
from app.tools.pipeline import SideEffectContext, prepare_side_effect_impls
from app.tools.ticket import ticket_create_tool
from app.tools.tools import calculate, get_weather
from app.utils.errors import build_error_info
from app.utils.logger import log_node

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME_GET_WEATHER,
            "description": "查询城市天气。适合处理北京、上海、广州等城市天气问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "要查询天气的城市名称，例如北京、上海、广州。",
                    }
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME_CALCULATE,
            "description": "计算数学表达式。适合处理加减乘除表达式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如 1+2 或 (3*5)-1。",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME_TICKET_CREATE,
            "description": (
                "创建工单。副作用工具：首次调用会返回需要确认的提示，"
                "客户端携带 confirmation_token 再次请求后才会真正落地。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "工单标题，简述用户诉求。",
                    },
                    "description": {
                        "type": "string",
                        "description": "工单详细描述，可留空。",
                    },
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME_MONITOR_QUERY_ERRORS,
            "description": (
                "查询指定 service 最近 30 分钟的 5xx/错误摘要，只读工具，"
                "适合排障类 workflow 的第一步。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": (
                            "service 名称（slug，如 payment-service、order-service）"
                        ),
                    }
                },
                "required": ["service"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_IMPLS = {
    TOOL_NAME_GET_WEATHER: get_weather,
    TOOL_NAME_CALCULATE: calculate,
    TOOL_NAME_TICKET_CREATE: ticket_create_tool,
    TOOL_NAME_MONITOR_QUERY_ERRORS: monitor_query_errors,
}


def format_single_tool_answer(tool_name: str, tool_output: str) -> str:
    if tool_name == TOOL_NAME_GET_WEATHER:
        return tool_output
    if tool_name == TOOL_NAME_CALCULATE:
        return f"计算结果是 {tool_output}。"
    return tool_output


def should_finalize_with_llm(message: str) -> bool:
    # 单工具、单意图问题直接模板化返回更划算；
    # 多工具或更复杂的组合问题，再交给 LLM 做整合。
    return any(keyword in message for keyword in TOOL_MULTI_INTENT_KEYWORDS)


def tool_agent_node(state: AgentState) -> AgentState:
    message = state["messages"][-1]["content"]
    finalize_with_llm = should_finalize_with_llm(message)
    on_delta, stream_state = build_answer_streamer(state, ROUTE_TOOL_AGENT)
    error_message = ""

    # Phase 1：基于 AuthContext.anonymous 过滤 side_effect 工具（第一层：匿名
    # 直接看不到这些工具）。然后再用 SideEffectContext 把 side_effect 工具的
    # impl 包装成"确认 + 幂等 + 超时 + 终态写入"的 pipeline（第二层）。
    auth = state["auth"]
    filtered_tools, filtered_impls = filter_tools_for_auth(
        TOOLS, TOOL_IMPLS, anonymous=auth.anonymous
    )

    side_ctx = SideEffectContext(
        auth=auth,
        session_id=state.get("session_id", "default"),
        request_id=state.get("request_id", ""),
        confirmation_token=state.get("confirmation_token", "") or "",
    )
    pipelined_impls = prepare_side_effect_impls(
        filtered_tools, filtered_impls, get_tool_metadata, side_ctx
    )

    try:
        tool_run: dict = {}
        replay_payload = None
        replay_error_code = ""
        confirmation_token = state.get("confirmation_token", "") or ""
        # Scheme A：有 token 时优先做"确定性重放"——从 token 解出 tool_name+args，
        # 直接调 pipeline 包装后的 impl，绕开 LLM。这样 step-2 不会因为 LLM 重
        # 述提示导致 args_hash 不匹配，idempotency 幂等闭环可以稳定成立。
        # 签名/过期失败：短路成与 pipeline 一致的错误文案，避免 LLM 自主改口。
        # secret 未配置：fall through 到 LLM 路径让原有提示文案（工具暂时不可用）
        # 生效，保持和 pipeline 对齐。
        if confirmation_token:
            try:
                replay_payload = decode_signed_payload(confirmation_token)
            except ConfirmationTokenError as exc:
                replay_error_code = exc.code
            except ConfirmationSecretMissing:
                replay_payload = None

        if replay_error_code:
            tool_run = {
                "tool_calls": [],
                "tool_results": [],
                "answer": f"确认 token 无效: {replay_error_code}",
            }
        elif replay_payload and replay_payload.tool_name in pipelined_impls:
            impl = pipelined_impls[replay_payload.tool_name]
            tool_output = impl(**replay_payload.args)
            tool_calls = [
                {
                    "id": "call_confirmation_replay",
                    "name": replay_payload.tool_name,
                    "arguments": dict(replay_payload.args),
                }
            ]
            tool_results = [
                {
                    "name": replay_payload.tool_name,
                    "output": tool_output,
                }
            ]
            tool_run = {
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "answer": "",
            }
        elif replay_payload is not None:
            # 安全门：token 解签通过，但 `tool_name` 在当前上下文里被
            # `filter_tools_for_auth` 过滤掉（匿名 × side_effect 最常见），
            # 或工具已经下架。这时**禁止**降级到 LLM 路径——否则 LLM 很可能
            # 走原生 function-calling 把该工具找回来，等于让一个"签名合法但无权"
            # 的 token 绕过鉴权。必须 fail-closed 给出明确错误码。
            tool_run = {
                "tool_calls": [],
                "tool_results": [],
                "answer": f"确认 token 无效: {ERR_TOKEN_AUTH_FORBIDDEN}",
            }
        else:
            tool_run = chat_with_tools(
                messages=[
                    {
                        "role": "system",
                        "content": TOOL_AGENT_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": message,
                    },
                ],
                tools=filtered_tools,
                tool_impls=pipelined_impls,
                finalize_with_llm=finalize_with_llm,
                on_delta=on_delta,
                profile=PROFILE_TOOL_CHAT,
            )
    except Exception as exc:
        tool_run = {}
        error_message = build_error_info(
            exc,
            stage=ROUTE_TOOL_AGENT,
            source="llm",
        )

    tool_calls = tool_run.get("tool_calls", [])
    tool_results = tool_run.get("tool_results", [])
    tool_type = (
        ",".join(call["name"] for call in tool_calls) if tool_calls else TOOL_TYPE_NONE
    )
    answer = tool_run.get("answer") or ""

    # 单工具场景直接模板化返回，省掉二次 LLM 整理成本。
    if not answer and len(tool_results) == 1:
        tool_name = tool_results[0]["name"]
        tool_output = tool_results[0]["output"]
        answer = format_single_tool_answer(tool_name, tool_output)

    # 多工具场景如果没有走二次 LLM，则退化为简单拼接。
    if not answer and tool_results:
        answer = " ".join(
            format_single_tool_answer(item["name"], item["output"])
            for item in tool_results
        )

    if error_message:
        answer = "工具调用暂时失败，请稍后再试。"

    if not answer:
        answer = "工具暂时无法处理这个问题。"

    # 若本轮出现 need_confirmation，pipeline 已经填好 pending_confirmation。
    # 覆盖 answer 为 pipeline 返回的提示文本，让客户端看到明确需求。
    pending_confirmation: dict = {}
    if side_ctx.pending_confirmation:
        pending_confirmation = dict(side_ctx.pending_confirmation)
        # answer 沿用 tool_result（pipeline 写入的提示），不再让兜底覆盖。
        # chat_with_tools 在 tool_results[0]["output"] 里就是提示文案。

    # debug_info / trace 侧只保留脱敏版 pending_confirmation：token 本体仅通过
    # 请求级 ContextVar 透出 API 响应（见 runtime_context.set_pending_confirmation_token）。
    # 这样即便 LangGraph 把节点返回的 state 拍进 checkpoint / LangSmith，也拿不到 token 原文。
    debug_pending = redact_pending_confirmation(pending_confirmation)
    if pending_confirmation.get("token"):
        set_pending_confirmation_token(pending_confirmation["token"])

    next_state: AgentState = {
        "tool_result": answer,
        "agent_outputs": {ROUTE_TOOL_AGENT: answer},
        "answer": answer,
        "tool_executions": list(side_ctx.executions),
        "pending_confirmation": debug_pending,
        "debug_info": {
            ROUTE_TOOL_AGENT: {
                "llm_profiles": {
                    PROFILE_TOOL_CHAT: get_profile_runtime_info(PROFILE_TOOL_CHAT),
                },
                "tool_type": tool_type,
                "tool_input": message,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "finalize_with_llm": finalize_with_llm,
                "streamed_answer": stream_state["used"],
                "error": error_message,
                "tool_result": answer,
                "tool_executions": list(side_ctx.executions),
                "pending_confirmation": debug_pending,
            }
        },
    }
    if stream_state["used"]:
        next_state["streamed_answer"] = True
    log_state = {**state, **next_state}

    log_node(
        ROUTE_TOOL_AGENT,
        log_state,
        extra={
            "toolType": tool_type,
            "toolInput": message,
            "toolCalls": tool_calls,
            "toolOutputs": tool_results,
            "finalizeWithLlm": finalize_with_llm,
            "error": error_message,
            "toolResult": answer,
            "toolExecutions": list(side_ctx.executions),
            "pendingConfirmation": debug_pending,
        },
    )
    return next_state

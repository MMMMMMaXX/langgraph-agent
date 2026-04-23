from app.constants.routes import ROUTE_TOOL_AGENT
from app.constants.model_profiles import PROFILE_TOOL_CHAT
from app.constants.tooling import (
    TOOL_MULTI_INTENT_KEYWORDS,
    TOOL_NAME_CALCULATE,
    TOOL_NAME_GET_WEATHER,
    TOOL_TYPE_NONE,
)
from app.state import AgentState
from app.tools.tools import get_weather, calculate
from app.streaming import build_answer_streamer
from app.utils.logger import log_node
from app.llm import chat_with_tools, get_profile_runtime_info
from app.prompts.tooling import TOOL_AGENT_SYSTEM_PROMPT
from app.utils.errors import build_error_info


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
]

TOOL_IMPLS = {
    TOOL_NAME_GET_WEATHER: get_weather,
    TOOL_NAME_CALCULATE: calculate,
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

    try:
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
            tools=TOOLS,
            tool_impls=TOOL_IMPLS,
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

    next_state: AgentState = {
        "tool_result": answer,
        "agent_outputs": {ROUTE_TOOL_AGENT: answer},
        "answer": answer,
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
            }
        }
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
        },
    )
    return next_state

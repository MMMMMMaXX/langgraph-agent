from typing import Annotated, Any, Literal

from typing_extensions import TypedDict


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """合并 LangGraph state 中的字典字段。

    用于 agent_outputs/debug_info/node_timings 这类“多节点逐步追加”的字段。
    节点只需要返回本轮新增的 key，LangGraph 会通过 reducer 自动合并，避免
    每个节点手写 dict(state.get(...)) 时遗漏防御性拷贝。
    """

    return {**(left or {}), **(right or {})}


class AgentState(TypedDict, total=False):
    request_id: str
    session_id: str
    debug: bool
    conversation_history_path: str
    stream_callback: Any
    streamed_answer: bool
    messages: list[dict]
    summary: str

    # supervisor 决策
    routes: list[Literal["rag_agent", "tool_agent", "chat_agent", "novel_script_agent"]]

    # 中间结果
    rewritten_query: str
    context: str
    tool_result: str
    agent_outputs: Annotated[dict, merge_dict]

    # 新增：vector memory 检索结果
    memory_hits: list[dict]
    node_timings: Annotated[dict, merge_dict]
    debug_info: Annotated[dict, merge_dict]

    # 最终输出
    answer: str

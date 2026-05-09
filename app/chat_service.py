from app.graph import graph
from app.llm import get_embedding_cache_stats, reset_embedding_cache
from app.runtime import SessionRuntime
from app.runtime.initial_state import create_initial_state as build_initial_state
from app.runtime_context import reset_stream_callback, set_stream_callback
from app.state import AgentState
from app.tracing import build_graph_trace_config, reset_model_call_index

_SESSION_RUNTIME = SessionRuntime()


def create_initial_state(session_id: str = "default") -> AgentState:
    """兼容旧入口：对外继续暴露 create_initial_state。"""

    return build_initial_state(session_id=session_id)


def run_chat_turn(state: AgentState, message: str) -> AgentState:
    user_message = message.strip()
    if not user_message:
        raise ValueError("message must not be empty")

    next_state: AgentState = {
        "request_id": state.get("request_id", ""),
        "session_id": state.get("session_id", "default"),
        "debug": state.get("debug", False),
        "conversation_history_path": state.get("conversation_history_path", ""),
        "streamed_answer": state.get("streamed_answer", False),
        "messages": list(state.get("messages", [])),
        "summary": state.get("summary", ""),
        "debug_info": dict(state.get("debug_info", {})),
        # Phase 1：AuthContext 沿着 run_chat_turn → graph 流下去。兼容入口
        # （chat_service.create_initial_state）已经写入默认匿名身份。
        "auth": state["auth"],
        # Phase 1 tool_safety：每轮请求新起确认 / 执行状态。
        "confirmation_token": state.get("confirmation_token", ""),
        "pending_confirmation": {},
        "tool_executions": [],
    }
    # 只有当前 state 里真的没有上下文时，才回退到 SessionRuntime 恢复。
    # 这样可以保持现有单进程热路径：同进程多轮对话仍优先复用 session cache，
    # 而进程重启后的冷恢复则自动落到 checkpoint。
    if not next_state["messages"] and not next_state["summary"]:
        snapshot = _SESSION_RUNTIME.load(next_state["session_id"], graph)
        if snapshot.messages:
            next_state["messages"] = list(snapshot.messages)
        if snapshot.summary:
            next_state["summary"] = snapshot.summary

    next_state["messages"].append({"role": "user", "content": user_message})
    graph_config = build_graph_trace_config(next_state, user_message)

    reset_model_call_index()
    reset_embedding_cache()
    callback_token = set_stream_callback(state.get("stream_callback"))
    try:
        result = graph.invoke(next_state, config=graph_config)
    finally:
        reset_stream_callback(callback_token)
    answer = result.get("answer", "").strip()
    updated_messages = _SESSION_RUNTIME.commit(
        session_id=next_state["session_id"],
        graph=graph,
        state=result,
        answer=answer,
    )

    debug_info = dict(result.get("debug_info", {}))
    debug_info["embedding_cache"] = get_embedding_cache_stats()

    return {
        "request_id": result.get("request_id", next_state["request_id"]),
        "session_id": result.get("session_id", next_state["session_id"]),
        "debug": result.get("debug", next_state["debug"]),
        "conversation_history_path": result.get(
            "conversation_history_path",
            next_state.get("conversation_history_path", ""),
        ),
        "messages": updated_messages,
        "summary": result.get("summary", ""),
        "routes": result.get("routes", []),
        "node_timings": result.get("node_timings", {}),
        "debug_info": debug_info,
        "answer": answer,
        "streamed_answer": result.get("streamed_answer", False),
        # Phase 1 tool_safety：把 side_effect pipeline 的输出透传给 API 层，
        # 由 chat_runner 决定是否暴露给客户端（pending_confirmation 会；
        # tool_executions 只在 debug=True 场景放到 debug_info 里）。
        "pending_confirmation": result.get("pending_confirmation") or {},
        "tool_executions": list(result.get("tool_executions") or []),
    }

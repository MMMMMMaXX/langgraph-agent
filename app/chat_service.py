from app.graph import graph
from app.llm import get_embedding_cache_stats, reset_embedding_cache
from app.runtime import SessionRuntime
from app.runtime.initial_state import create_initial_state as build_initial_state
from app.runtime_context import (
    begin_pending_confirmation_scope,
    get_pending_confirmation_token,
    reset_pending_confirmation_token,
    reset_stream_callback,
    set_stream_callback,
)
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
    # 每次 turn 新建一个"confirmation token 原文"上下文；节点通过 append 向
    # holder 里塞 token，请求出口再读出注入 API 响应，避免 token 落进
    # AgentState / checkpoint / LangSmith。
    # 注意：LangGraph 会在子 Context 里跑节点，单纯 ContextVar.set(str) 写入
    # 不会回传父上下文；这里换成 list holder 让 append 的效果跨上下文可见。
    pending_token_ctx = begin_pending_confirmation_scope()
    try:
        result = graph.invoke(next_state, config=graph_config)
        pending_token_plain = get_pending_confirmation_token()
    finally:
        reset_stream_callback(callback_token)
        reset_pending_confirmation_token(pending_token_ctx)
    answer = result.get("answer", "").strip()
    updated_messages = _SESSION_RUNTIME.commit(
        session_id=next_state["session_id"],
        graph=graph,
        state=result,
        answer=answer,
    )

    debug_info = dict(result.get("debug_info", {}))
    debug_info["embedding_cache"] = get_embedding_cache_stats()

    # Graph state 里的 pending_confirmation 是脱敏版（token_present 布尔 + 元
    # 数据），API 出口再把 ContextVar 里的原文 token 贴回来。token 从此**只**
    # 出现在 chat_runner 构造的 HTTP 响应体里，不会进 checkpoint/trace。
    pending_redacted = dict(result.get("pending_confirmation") or {})
    if pending_redacted and pending_token_plain:
        pending_redacted["token"] = pending_token_plain

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
        "pending_confirmation": pending_redacted,
        "tool_executions": list(result.get("tool_executions") or []),
    }

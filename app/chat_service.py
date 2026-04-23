from app.graph import graph
from app.llm import get_embedding_cache_stats, reset_embedding_cache
from app.state import AgentState
from app.tracing import build_graph_trace_config, reset_model_call_index


def create_initial_state(session_id: str = "default") -> AgentState:
    return {
        "session_id": session_id,
        "debug": False,
        "messages": [],
        "summary": "",
    }


def run_chat_turn(state: AgentState, message: str) -> AgentState:
    user_message = message.strip()
    if not user_message:
        raise ValueError("message must not be empty")

    next_state: AgentState = {
        "request_id": state.get("request_id", ""),
        "session_id": state.get("session_id", "default"),
        "debug": state.get("debug", False),
        "conversation_history_path": state.get("conversation_history_path", ""),
        "stream_callback": state.get("stream_callback"),
        "streamed_answer": state.get("streamed_answer", False),
        "messages": list(state.get("messages", [])),
        "summary": state.get("summary", ""),
        "debug_info": dict(state.get("debug_info", {})),
    }
    next_state["messages"].append({"role": "user", "content": user_message})

    reset_model_call_index()
    reset_embedding_cache()
    result = graph.invoke(
        next_state,
        config=build_graph_trace_config(next_state, user_message),
    )
    answer = result.get("answer", "").strip()

    updated_messages = list(result.get("messages", []))
    updated_messages.append({"role": "assistant", "content": answer})

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
    }

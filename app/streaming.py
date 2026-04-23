from collections.abc import Callable

from app.state import AgentState


def build_answer_streamer(
    state: AgentState, node_name: str
) -> tuple[Callable[[str], None] | None, dict[str, bool]]:
    stream_callback = state.get("stream_callback")
    request_id = state.get("request_id", "")
    session_id = state.get("session_id", "default")
    stream_state = {"used": False}

    if not callable(stream_callback):
        return None, stream_state

    # 这里做的是“节点输出 -> SSE chunk 事件”的桥接，
    # LLM 层只负责产生 delta，不应该感知 HTTP / SSE 协议细节。
    def on_delta(delta: str) -> None:
        if not delta:
            return
        stream_state["used"] = True
        stream_callback(
            "chunk",
            {
                "request_id": request_id,
                "session_id": session_id,
                "node": node_name,
                "delta": delta,
            },
        )

    return on_delta, stream_state

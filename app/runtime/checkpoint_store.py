"""LangGraph checkpoint 门面。

这层把 `graph.get_state()` / `graph.update_state()` 细节封起来，避免
chat_service 直接感知 checkpoint 读写协议。
"""

from __future__ import annotations

from typing import Any

from app.constants.routes import NODE_MEMORY
from app.constants.runtime import (
    RUNTIME_RESTORE_FROM_CHECKPOINT,
    RUNTIME_RESTORE_FROM_EMPTY,
)
from app.runtime.snapshot import ConversationSnapshot


def build_checkpoint_graph_config(session_id: str) -> dict[str, Any]:
    """构造只包含 thread_id 的 checkpoint config。

    checkpoint 恢复/提交只依赖 thread_id，把 tracing metadata 留给真正的 graph.invoke。
    这样 runtime 层只关心“会话连续性”，不耦合请求级 trace 细节。
    """

    return {"configurable": {"thread_id": session_id}}


def load_checkpoint_snapshot(graph: Any, session_id: str) -> ConversationSnapshot:
    """从 LangGraph checkpoint 恢复会话快照。

    注意：checkpoint 只是一种恢复来源，不保证一定有值。
    没命中时返回 empty snapshot，而不是抛错，让调用方可以无脑走后续流程。
    """

    checkpoint_snapshot = graph.get_state(build_checkpoint_graph_config(session_id))
    checkpoint_values = checkpoint_snapshot.values if checkpoint_snapshot else {}
    messages = list(checkpoint_values.get("messages", []))
    summary = checkpoint_values.get("summary", "")

    if not messages and not summary:
        return ConversationSnapshot(
            session_id=session_id,
            thread_id=session_id,
            restored_from=RUNTIME_RESTORE_FROM_EMPTY,
            has_checkpoint_state=False,
        )

    return ConversationSnapshot(
        session_id=session_id,
        thread_id=session_id,
        messages=messages,
        summary=summary,
        restored_from=RUNTIME_RESTORE_FROM_CHECKPOINT,
        has_checkpoint_state=True,
    )


def persist_final_checkpoint_state(
    graph: Any,
    session_id: str,
    state: dict[str, Any],
    answer: str,
) -> list[dict[str, Any]]:
    """把本轮最终 assistant 回答补写回 checkpoint。

    LangGraph 自动保存的是图执行过程中的 state，但当前项目的最终 assistant 消息
    仍是在 graph.invoke 返回后补到 messages 里，所以这里统一负责把最终态写回。

    返回值是“已经补齐 assistant 消息后的 messages”，供上层继续复用，避免重复拼接。
    """

    updated_messages = list(state.get("messages", []))
    # 这里故意不判断 answer 是否为空字符串。
    # 原有 chat_service 的语义是：一轮 graph 结束后，总会补一条 assistant message，
    # 哪怕内容为空，也要保证 messages 里的 user/assistant 轮次闭环不被破坏。
    updated_messages.append({"role": "assistant", "content": answer})

    graph.update_state(
        build_checkpoint_graph_config(session_id),
        {
            "messages": updated_messages,
            "summary": state.get("summary", ""),
        },
        as_node=NODE_MEMORY,
    )
    return updated_messages

"""app.nodes.memory 单测：

- 三个纯函数分支（should_refresh_summary / should_skip_summary_refresh /
  should_skip_history_store）
- memory_node 的编排：LLM summary 失败 / vector 写入失败 / history 写入失败 /
  duplicate / message pruning

所有外部 I/O（summarize_messages / add_memory_item / append_history_event）
都在 `app.nodes.memory` 模块命名空间上 monkeypatch 掉。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import app.nodes.memory as mem_mod
from app.config import MEMORY_CONFIG
from app.constants.policies import (
    SKIP_REASON_DUPLICATE,
    SKIP_REASON_EMPTY_MESSAGE,
    SKIP_REASON_META_QUERY,
    SKIP_REASON_RAG_DOC_HIT,
)
from app.constants.routes import (
    NODE_MEMORY,
    ROUTE_CHAT_AGENT,
    ROUTE_RAG_AGENT,
)
from app.llm import LLMCallError
from app.nodes.memory import (
    memory_node,
    should_refresh_summary,
    should_skip_history_store,
    should_skip_summary_refresh,
)


# ---------------------------------------------------------------------------
# should_refresh_summary
# ---------------------------------------------------------------------------


def test_should_refresh_summary_false_when_too_few_messages() -> None:
    assert should_refresh_summary([], "q") is False
    assert should_refresh_summary([{"role": "user", "content": "hi"}], "q") is False


def test_should_refresh_summary_true_when_over_trigger() -> None:
    # summary_trigger=6 → 7 条触发
    msgs = [{"role": "user", "content": str(i)} for i in range(7)]
    assert should_refresh_summary(msgs, "随便什么查询") is True


def test_should_refresh_summary_true_when_meta_keyword_in_query() -> None:
    """即使对话很短，用户明确问"刚刚/总结"也要刷新 summary。"""
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    assert should_refresh_summary(msgs, "帮我总结下刚才的对话") is True


def test_should_refresh_summary_false_when_short_and_no_meta() -> None:
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    assert should_refresh_summary(msgs, "北京天气怎么样") is False


# ---------------------------------------------------------------------------
# should_skip_summary_refresh
# ---------------------------------------------------------------------------


def test_should_skip_summary_refresh_true_for_chat_agent_meta_query() -> None:
    state = {"routes": [ROUTE_CHAT_AGENT]}
    assert should_skip_summary_refresh(state, "刚才我问了啥") is True


def test_should_skip_summary_refresh_false_for_rag_agent_even_with_meta() -> None:
    """走到 rag_agent 时，即便问了 meta 也不能跳过 summary 刷新。"""
    state = {"routes": [ROUTE_RAG_AGENT]}
    assert should_skip_summary_refresh(state, "刚才我问了啥") is False


def test_should_skip_summary_refresh_false_for_chat_agent_normal_query() -> None:
    state = {"routes": [ROUTE_CHAT_AGENT]}
    assert should_skip_summary_refresh(state, "你好") is False


# ---------------------------------------------------------------------------
# should_skip_history_store
# ---------------------------------------------------------------------------


def test_should_skip_history_empty_message() -> None:
    skipped, reason = should_skip_history_store("   ", "anything")
    assert skipped is True
    assert reason == SKIP_REASON_EMPTY_MESSAGE


def test_should_skip_history_meta_query() -> None:
    skipped, reason = should_skip_history_store("总结一下", "总结一下？")
    assert skipped is True
    assert reason == SKIP_REASON_META_QUERY


def test_should_skip_history_normal_message_not_skipped() -> None:
    skipped, reason = should_skip_history_store("北京天气", "北京天气怎么样？")
    assert skipped is False
    assert reason == ""


# ---------------------------------------------------------------------------
# memory_node orchestration
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_io(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """打桩 memory_node 的三个外部调用：summarize / add_memory / append_history。

    测试可改 harness 切换失败场景：
        harness["summary_fn"] = lambda *_, **__: raise ...
        harness["add_memory_fn"] = ...
        harness["history_result"] = {"skipped_duplicate": True, ...}
    """

    harness: dict[str, Any] = {
        "summary_fn": lambda old, recent: "new summary",
        "add_memory_fn": lambda *args, **kwargs: None,
        "history_result": {
            "skipped_duplicate": False,
            "rewritten_query": "",
            "user_message": "",
        },
        "history_error": None,
        "add_memory_calls": 0,
        "append_history_calls": 0,
        "summarize_calls": 0,
    }

    def fake_summarize(old_summary: str, recent: list) -> str:
        harness["summarize_calls"] += 1
        return harness["summary_fn"](old_summary, recent)

    def fake_add_memory(*args, **kwargs) -> None:
        harness["add_memory_calls"] += 1
        harness["add_memory_fn"](*args, **kwargs)

    def fake_append_history(**kwargs) -> dict:
        harness["append_history_calls"] += 1
        if harness["history_error"] is not None:
            raise harness["history_error"]
        result = dict(harness["history_result"])
        # 让返回结构映射实际 kwargs，方便断言
        result.setdefault("rewritten_query", kwargs.get("rewritten_query", ""))
        result.setdefault("user_message", kwargs.get("user_message", ""))
        return result

    monkeypatch.setattr(mem_mod, "summarize_messages", fake_summarize)
    monkeypatch.setattr(mem_mod, "add_memory_item", fake_add_memory)
    monkeypatch.setattr(mem_mod, "append_history_event", fake_append_history)
    yield harness


def _state_with_answer(
    *,
    messages: list[dict] | None = None,
    answer: str = "北京今天晴天，气温25度，适合外出活动。",
    routes: list[str] | None = None,
    rewritten_query: str = "北京今天天气怎么样？",
    summary: str = "",
) -> dict:
    """造一个会触发 vector write 的典型 state。

    答案长度 > 8 字 + 非 "资料不足"，decide_memory_write 会放行。
    默认走 rag_agent 路由但不标 doc_used，避免触发 RAG_DOC_HIT 跳过。
    """
    return {
        "session_id": "s1",
        "messages": messages or [{"role": "user", "content": "北京天气"}],
        "summary": summary,
        "answer": answer,
        "rewritten_query": rewritten_query,
        "routes": routes or [ROUTE_RAG_AGENT],
        "debug_info": {ROUTE_RAG_AGENT: {"doc_used": False}},
    }


def test_memory_node_happy_path_writes_vector_and_history(
    memory_io: dict[str, Any],
) -> None:
    state = _state_with_answer()
    new_state = memory_node(state)

    debug = new_state["debug_info"][NODE_MEMORY]
    assert debug["stored_to_vector"] is True
    assert debug["stored_to_history"] is True
    assert debug["errors"] == []
    # 默认短对话不触发 summary 刷新
    assert debug["refreshed_summary"] is False
    # 三个 sub timing 都记录了
    assert set(debug["sub_timings_ms"].keys()) == {
        "summaryRefresh",
        "vectorStore",
        "historyStore",
        "messagePrune",
    }
    # 实际 I/O 次数
    assert memory_io["add_memory_calls"] == 1
    assert memory_io["append_history_calls"] == 1


def test_memory_node_skips_vector_for_rag_doc_hit_but_keeps_history(
    memory_io: dict[str, Any],
) -> None:
    """RAG 命中文档时，docs 是事实源；只写 history，不反写 semantic memory。"""
    state = _state_with_answer(
        answer="WAI-ARIA 和虚拟列表是不同技术，分别关注无障碍和渲染性能。",
        rewritten_query="WAI-ARIA 和虚拟列表有什么区别？",
    )
    state["debug_info"][ROUTE_RAG_AGENT] = {
        "doc_used": True,
        "query_type": "comparison",
        "answer_strategy": "comparison",
    }

    new_state = memory_node(state)

    debug = new_state["debug_info"][NODE_MEMORY]
    assert debug["stored_to_vector"] is False
    assert debug["skipped_vector_store"] is True
    assert debug["vector_store_skip_reason"] == SKIP_REASON_RAG_DOC_HIT
    assert debug["stored_to_history"] is True
    assert memory_io["add_memory_calls"] == 0
    assert memory_io["append_history_calls"] == 1


def test_memory_node_skips_summary_for_meta_chat_query(
    memory_io: dict[str, Any],
) -> None:
    """chat_agent + meta query → skipped_summary_refresh=True，不调 summarize。"""
    state = _state_with_answer(
        messages=[{"role": "user", "content": "总结一下刚才"}],
        rewritten_query="总结一下刚才",
        answer="刚才我们讨论了天气",
        routes=[ROUTE_CHAT_AGENT],
    )
    new_state = memory_node(state)
    debug = new_state["debug_info"][NODE_MEMORY]
    assert debug["skipped_summary_refresh"] is True
    assert debug["refreshed_summary"] is False
    assert memory_io["summarize_calls"] == 0


def test_memory_node_refreshes_summary_when_triggered(
    memory_io: dict[str, Any],
) -> None:
    """消息数 > summary_trigger → 触发刷新。"""
    msgs = [{"role": "user", "content": f"msg-{i}"} for i in range(8)]
    state = _state_with_answer(messages=msgs)

    new_state = memory_node(state)
    debug = new_state["debug_info"][NODE_MEMORY]
    assert debug["refreshed_summary"] is True
    assert new_state["summary"] == "new summary"
    assert memory_io["summarize_calls"] == 1


def test_memory_node_summarize_llm_error_recorded(
    memory_io: dict[str, Any],
) -> None:
    """summarize LLM 失败 → 记录 error；summary 保持原值；链路不中断。"""

    def raise_llm_err(old, recent):
        raise LLMCallError("llm_error", "boom")

    memory_io["summary_fn"] = raise_llm_err
    msgs = [{"role": "user", "content": f"msg-{i}"} for i in range(8)]
    old_summary = "old summary"
    state = _state_with_answer(messages=msgs, summary=old_summary)

    new_state = memory_node(state)
    debug = new_state["debug_info"][NODE_MEMORY]
    # summary 回滚到原值
    assert new_state["summary"] == old_summary
    assert debug["refreshed_summary"] is False
    assert len(debug["errors"]) >= 1
    # 后续 vector/history 仍正常执行
    assert memory_io["add_memory_calls"] == 1
    assert memory_io["append_history_calls"] == 1


def test_memory_node_vector_write_error_does_not_block_history(
    memory_io: dict[str, Any],
) -> None:
    """add_memory_item 抛异常 → 记录 error，stored_to_vector=False，history 继续。"""

    def raise_storage(*_args, **_kwargs):
        raise RuntimeError("chroma full")

    memory_io["add_memory_fn"] = raise_storage

    state = _state_with_answer()
    new_state = memory_node(state)

    debug = new_state["debug_info"][NODE_MEMORY]
    assert debug["stored_to_vector"] is False
    assert len(debug["errors"]) >= 1
    # history 仍被尝试
    assert memory_io["append_history_calls"] == 1
    assert debug["stored_to_history"] is True


def test_memory_node_history_append_error_recorded(
    memory_io: dict[str, Any],
) -> None:
    """append_history_event 抛异常 → 记录 error，stored_to_history=False。"""
    memory_io["history_error"] = RuntimeError("sqlite locked")

    state = _state_with_answer()
    new_state = memory_node(state)

    debug = new_state["debug_info"][NODE_MEMORY]
    assert debug["stored_to_history"] is False
    assert len(debug["errors"]) >= 1
    # vector 已成功
    assert debug["stored_to_vector"] is True


def test_memory_node_history_duplicate_marks_skip(
    memory_io: dict[str, Any],
) -> None:
    """append_history 返回 skipped_duplicate=True → 标记 skipped_history_store。"""
    memory_io["history_result"] = {
        "skipped_duplicate": True,
        "rewritten_query": "北京？",
        "user_message": "北京天气",
    }

    state = _state_with_answer()
    new_state = memory_node(state)

    debug = new_state["debug_info"][NODE_MEMORY]
    assert debug["stored_to_history"] is False
    assert debug["skipped_history_store"] is True
    assert debug["history_store_skip_reason"] == SKIP_REASON_DUPLICATE


def test_memory_node_prunes_messages_when_too_long(
    memory_io: dict[str, Any],
) -> None:
    """messages > summary_trigger → 裁剪到 max_recent_messages。"""
    trigger = MEMORY_CONFIG.summary_trigger
    max_recent = MEMORY_CONFIG.max_recent_messages
    msgs = [{"role": "user", "content": f"msg-{i}"} for i in range(trigger + 2)]
    state = _state_with_answer(messages=msgs)

    new_state = memory_node(state)

    # 新 messages 被裁到 max_recent_messages
    assert len(new_state["messages"]) == max_recent
    # 裁掉的是开头几条，保留的是尾部
    assert new_state["messages"][-1]["content"] == f"msg-{trigger + 1}"


def test_memory_node_skips_history_for_empty_message(
    memory_io: dict[str, Any],
) -> None:
    """用户消息为空 → should_skip_history_store True，完全不调 append。"""
    state = _state_with_answer(
        messages=[{"role": "user", "content": "   "}],
        rewritten_query="",
        answer="auto",
    )
    new_state = memory_node(state)

    debug = new_state["debug_info"][NODE_MEMORY]
    assert debug["skipped_history_store"] is True
    assert debug["history_store_skip_reason"] == SKIP_REASON_EMPTY_MESSAGE
    assert memory_io["append_history_calls"] == 0

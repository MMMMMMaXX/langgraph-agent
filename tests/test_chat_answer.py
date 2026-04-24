"""`app.agents.chat.answer` 两个 LLM 薄包装的单测。

这两个函数是 chat_agent 真正和 LLM 对话的出口，不接 Chroma / 历史后端。
测试关注三件事：
1. system / user prompt 拼装是否按模板注入（task / summary / memory_context / facts）
2. profile 选择是否正确（QA → default_chat，summary → summary）
3. 流式 on_delta 是否被正确透传给 chat()

上游节点复杂度再高，只要这层契约稳，换实现不怕回归。
"""

from __future__ import annotations

from app.agents.chat.answer import generate_answer, generate_summary_answer
from app.constants.model_profiles import PROFILE_DEFAULT_CHAT, PROFILE_SUMMARY

# ----------------------------- generate_answer -----------------------------


def test_generate_answer_injects_all_context_fields(llm_stub) -> None:
    llm_stub.set_response("final-answer")

    out = generate_answer(
        message="北京天气怎么样",
        summary="之前聊过上海",
        memory_context="memo: 用户在北京",
        facts_text="facts: 城市=北京",
        task="qa",
    )

    # 返回值直接来自桩
    assert out == "final-answer"

    # 只应调用一次 LLM
    assert len(llm_stub.calls) == 1
    call = llm_stub.calls[0]

    # profile：QA 走 default_chat
    assert call["profile"] == PROFILE_DEFAULT_CHAT

    # messages 结构：system + user 两条
    messages = call["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    # system 里带 task 类型
    assert "qa" in messages[0]["content"]

    # user 里四块内容都要注入
    user_content = messages[1]["content"]
    assert "memo: 用户在北京" in user_content
    assert "facts: 城市=北京" in user_content
    assert "之前聊过上海" in user_content
    assert "北京天气怎么样" in user_content


def test_generate_answer_empty_fields_render_as_no(llm_stub) -> None:
    # 不能因为 summary / memory_context 为空就崩；模板会把空值渲染成 "无"
    generate_answer(
        message="你好",
        summary="",
        memory_context="",
        facts_text="",
        task="qa",
    )

    user_content = llm_stub.calls[0]["messages"][1]["content"]
    # 两处 "无" 分别对应 memory_context 和 summary
    assert user_content.count("无") >= 2
    assert "你好" in user_content


def test_generate_answer_forwards_on_delta(llm_stub) -> None:
    # 设置返回值 "hello-stream"；桩内部会用 on_delta 把它一次性吐出去
    llm_stub.set_response("hello-stream")

    received: list[str] = []

    generate_answer(
        message="q",
        summary="",
        memory_context="",
        facts_text="",
        task="qa",
        on_delta=lambda d: received.append(d),
    )

    assert received == ["hello-stream"]


def test_generate_answer_task_type_flows_into_system_prompt(llm_stub) -> None:
    # task=recall 时 system prompt 里应当出现 recall 字样（防止上游传错 task 不被察觉）
    generate_answer(
        message="q",
        summary="",
        memory_context="",
        facts_text="",
        task="recall",
    )

    system_content = llm_stub.calls[0]["messages"][0]["content"]
    assert "recall" in system_content


# -------------------------- generate_summary_answer --------------------------


def test_generate_summary_answer_uses_summary_profile(llm_stub) -> None:
    llm_stub.set_response("summary-answer")

    out = generate_summary_answer(
        message="我们聊过什么",
        summary="聊过天气和北京",
    )

    assert out == "summary-answer"
    call = llm_stub.calls[0]
    # 关键契约：summary 路径必须切到 summary profile，否则成本 / 模型选型就错了
    assert call["profile"] == PROFILE_SUMMARY

    messages = call["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    user_content = messages[1]["content"]
    assert "聊过天气和北京" in user_content
    assert "我们聊过什么" in user_content


def test_generate_summary_answer_empty_summary_renders_no(llm_stub) -> None:
    generate_summary_answer(message="q", summary="")

    user_content = llm_stub.calls[0]["messages"][1]["content"]
    assert "无" in user_content


def test_generate_summary_answer_forwards_on_delta(llm_stub) -> None:
    llm_stub.set_response("piece")
    received: list[str] = []

    generate_summary_answer(
        message="q",
        summary="s",
        on_delta=lambda d: received.append(d),
    )

    assert received == ["piece"]

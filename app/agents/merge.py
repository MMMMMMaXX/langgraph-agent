from app.constants.model_profiles import PROFILE_DEFAULT_CHAT
from app.constants.policies import INSUFFICIENT_KNOWLEDGE_ANSWER
from app.constants.routes import NODE_MERGE
from app.llm import LLMCallError, chat, get_profile_runtime_info
from app.prompts.merge import MERGE_SYSTEM_PROMPT, build_merge_user_prompt
from app.state import AgentState
from app.streaming import build_answer_streamer
from app.utils.logger import log_node, preview


def merge_node(state: AgentState) -> AgentState:
    outputs = state.get("agent_outputs", {})
    message = state["messages"][-1]["content"]
    on_delta, stream_state = build_answer_streamer(state, NODE_MERGE)
    merge_error = ""

    if not outputs:
        answer = INSUFFICIENT_KNOWLEDGE_ANSWER
    elif len(outputs) == 1:
        answer = list(outputs.values())[0]
    else:
        merged_input = "\n".join([f"{k}: {v}" for k, v in outputs.items()])

        try:
            answer = chat(
                [
                    {
                        "role": "system",
                        "content": MERGE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": build_merge_user_prompt(message, merged_input),
                    },
                ],
                on_delta=on_delta,
                profile=PROFILE_DEFAULT_CHAT,
            )
        except LLMCallError as exc:
            # 已知的 LLM 调用失败（超时、鉴权、限流等）：降级拼接各 agent 输出
            merge_error = f"llm_call_error: {exc}"
            answer = " ".join(
                str(v).strip() for v in outputs.values() if str(v).strip()
            )
            if not answer:
                answer = INSUFFICIENT_KNOWLEDGE_ANSWER
        except Exception as exc:
            # 兜底：prompt 构造异常、编码错误、stream 回调内部异常等非预期情况。
            # 不能让 merge 节点抛出，否则整个 graph 失败；保留 repr 方便排查。
            merge_error = f"unexpected_error: {exc!r}"
            answer = " ".join(
                str(v).strip() for v in outputs.values() if str(v).strip()
            )
            if not answer:
                answer = INSUFFICIENT_KNOWLEDGE_ANSWER

    next_state: AgentState = {
        "answer": answer,
        "debug_info": {
            NODE_MERGE: {
                "llm_profiles": {
                    "merge_answer": get_profile_runtime_info(PROFILE_DEFAULT_CHAT),
                },
                "agent_outputs": state.get("agent_outputs", {}),
                "final_answer_preview": preview(answer, 160),
                "streamed_answer": stream_state["used"],
                "error": merge_error,
            }
        },
    }
    if stream_state["used"]:
        next_state["streamed_answer"] = True
    log_state = {**state, **next_state}

    log_node(
        NODE_MERGE,
        log_state,
        extra={
            "agentOutputs": state.get("agent_outputs", {}),
            "finalAnswerPreview": preview(answer, 160),
            "error": merge_error,
        },
    )
    return next_state

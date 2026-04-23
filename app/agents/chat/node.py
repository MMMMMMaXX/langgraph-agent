from app.agents.chat.debug import (
    build_chat_debug_payload,
    build_chat_log_extra,
    build_retrieval_debug,
)
from app.agents.chat.flow import generate_chat_answer
from app.agents.chat.intent import (
    classify_chat_operator,
)
from app.agents.chat.memory_retrieval import (
    build_memory_facts,
    build_structured_facts_text,
    prepare_memory_hits,
)
from app.constants.policies import (
    HISTORY_POLICY_NONE,
)
from app.constants.routes import ROUTE_CHAT_AGENT
from app.state import AgentState
from app.streaming import build_answer_streamer
from app.utils.logger import log_node, now_ms
from app.memory.conversation_history import (
    resolve_history_backend,
)


def chat_agent_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "default")
    message = state["messages"][-1]["content"]
    messages = state.get("messages", [])
    summary = state.get("summary", "")
    conversation_history_path = state.get("conversation_history_path", "")
    conversation_history_backend = resolve_history_backend(
        history_path=conversation_history_path or None
    )
    sub_timings_ms: dict[str, float] = {}

    memory_prepare_started_at_ms = now_ms()
    memory_hits, task, memory_lookup_policy, memory_before_rerank, errors = prepare_memory_hits(
        message, session_id
    )
    sub_timings_ms["memoryPrepare"] = round(now_ms() - memory_prepare_started_at_ms, 2)

    operator_classify_started_at_ms = now_ms()
    operator = classify_chat_operator(message)
    sub_timings_ms["operatorClassify"] = round(
        now_ms() - operator_classify_started_at_ms, 2
    )

    context_build_started_at_ms = now_ms()
    memory_context = "\n\n".join([m["content"] for m in memory_hits])
    facts = build_memory_facts(memory_hits)
    facts_text = build_structured_facts_text(facts)
    sub_timings_ms["contextBuild"] = round(now_ms() - context_build_started_at_ms, 2)
    on_delta, stream_state = build_answer_streamer(state, ROUTE_CHAT_AGENT)

    answer_result = generate_chat_answer(
        operator=operator,
        task=task,
        memory_lookup_policy=memory_lookup_policy,
        memory_hits=memory_hits,
        message=message,
        messages=messages,
        session_id=session_id,
        summary=summary,
        conversation_history_path=conversation_history_path,
        memory_context=memory_context,
        facts_text=facts_text,
        on_delta=on_delta,
    )
    sub_timings_ms.update(answer_result.sub_timings_ms)
    errors.extend(answer_result.errors)

    retrieval_debug = build_retrieval_debug(
        session_id=session_id,
        memory_lookup_policy=memory_lookup_policy,
        history_lookup_policy=answer_result.history_lookup_policy,
        conversation_history_backend=conversation_history_backend,
        conversation_history_path=conversation_history_path,
        memory_before_rerank=memory_before_rerank,
        memory_hits=memory_hits,
        working_memory_items=answer_result.working_memory_items,
        history_events=answer_result.history_events,
        history_items=answer_result.history_items,
        used_memory=answer_result.used_memory,
        used_history=answer_result.used_history,
    )
    next_state: AgentState = {
        "agent_outputs": {ROUTE_CHAT_AGENT: answer_result.answer},
        "answer": answer_result.answer,
        "debug_info": {
            ROUTE_CHAT_AGENT: build_chat_debug_payload(
                task=task,
                memory_lookup_policy=memory_lookup_policy,
                history_lookup_policy=answer_result.history_lookup_policy,
                used_memory=answer_result.used_memory,
                used_history=answer_result.used_history,
                used_summary=answer_result.used_summary,
                streamed_answer=stream_state["used"],
                memory_before_rerank=memory_before_rerank,
                memory_hits=memory_hits,
                history_events=answer_result.history_events,
                retrieval_debug=retrieval_debug,
                facts=facts,
                sub_timings_ms=sub_timings_ms,
                errors=errors,
            )
        },
    }
    if stream_state["used"]:
        next_state["streamed_answer"] = True
    log_state = {**state, **next_state}

    log_node(
        ROUTE_CHAT_AGENT,
        log_state,
        extra=build_chat_log_extra(
            task=task,
            memory_lookup_policy=memory_lookup_policy,
            history_lookup_policy=answer_result.history_lookup_policy,
            used_memory=answer_result.used_memory,
            used_history=answer_result.used_history,
            used_summary=answer_result.used_summary,
            memory_before_rerank=memory_before_rerank,
            memory_hits=memory_hits,
            history_events=answer_result.history_events,
            retrieval_debug=retrieval_debug,
            facts=facts,
            sub_timings_ms=sub_timings_ms,
            errors=errors,
        ),
    )
    return next_state

from app.agents.novel_script.constants import (
    DEFAULT_TARGET_SCENE_COUNT,
    FAST_MODE_SOURCE_CHARS,
    FAST_MODE_TARGET_SCENE_COUNT,
    MAX_SCENE_REWRITE_ATTEMPTS,
    NOVEL_SCRIPT_NO_REVIEW_FINALIZE_BUFFER,
    NOVEL_SCRIPT_REVIEW_CYCLE_ITERATIONS,
    NOVEL_SCRIPT_SETUP_ITERATIONS,
    REVIEW_ENABLED_MAX_SOURCE_CHARS,
)
from app.agents.novel_script.graph import novel_script_graph
from app.agents.novel_script.state import NovelScriptState
from app.agents.novel_script.tools import CHAPTER_TITLE_RE
from app.constants.model_profiles import (
    PROFILE_CREATIVE_PLANNER,
    PROFILE_CREATIVE_REVIEW,
    PROFILE_CREATIVE_WRITE,
)
from app.constants.policies import INSUFFICIENT_KNOWLEDGE_ANSWER
from app.constants.routes import ROUTE_NOVEL_SCRIPT_AGENT
from app.llm import get_profile_runtime_info
from app.runtime_context import get_stream_callback
from app.state import AgentState
from app.utils.logger import log_node, preview


def looks_like_script_task(message: str) -> bool:
    # 先用轻量规则识别“创作型改编”请求，
    # 这样可以避免每次都让 supervisor 走 LLM planner。
    keywords = [
        "改成剧本",
        "改编成剧本",
        "改编成短剧",
        "转换为剧本",
        "生成剧本",
        "写成剧本",
        "写成短剧",
        "场景剧本",
        "对白脚本",
    ]
    return any(keyword in message for keyword in keywords)


def extract_source_text(message: str) -> str:
    # 第一版先做最简单的文本抽取：
    # 用户通常会写成“把下面小说改成剧本：<正文>”
    # 或者直接换行贴原文，这里优先把指令头和正文拆开。
    separators = ["：", "\n\n", "\n"]
    for separator in separators:
        if separator in message:
            _, content = message.split(separator, 1)
            content = content.strip()
            if content:
                return content
    return message.strip()


def count_chapter_boundaries(source_text: str) -> int:
    # 章节标题是比“按字数估算场景数”更强的结构信号。
    # 如果用户贴来的小说已经写了“第1章 / 第一章”，
    # 我们就应该优先尊重作者给出的章节边界。
    count = 0
    for line in source_text.splitlines():
        if CHAPTER_TITLE_RE.match(line.strip()):
            count += 1
    return count


def calculate_max_iterations(target_scene_count: int, enable_review: bool) -> int:
    """计算 novel_script 子图的最大动作轮次。

    一轮动作对应一次实际 tool 执行，例如 split / facts / write / review。
    review 闭环至少需要：
    - 初稿：target_scene_count 次 write_script_scene
    - 首审：1 次 review_script
    - 返修：每个问题场景最多 MAX_SCENE_REWRITE_ATTEMPTS 次 write_script_scene
    - 复审：1 次 review_script，确认返修后的最新版草稿已经被检查过
    """
    if not enable_review:
        return (
            NOVEL_SCRIPT_SETUP_ITERATIONS
            + target_scene_count
            + NOVEL_SCRIPT_NO_REVIEW_FINALIZE_BUFFER
        )

    return (
        NOVEL_SCRIPT_SETUP_ITERATIONS
        + target_scene_count
        + target_scene_count * MAX_SCENE_REWRITE_ATTEMPTS
        + NOVEL_SCRIPT_REVIEW_CYCLE_ITERATIONS
    )


def novel_script_agent_node(state: AgentState) -> AgentState:
    message = state["messages"][-1]["content"].strip()
    source_text = extract_source_text(message)
    source_text_len = len(source_text)
    chapter_count = count_chapter_boundaries(source_text)

    # 长文本创作链路要控制“场景数 * LLM 调用次数”。
    # 否则在评测或真实线上请求里，很容易因为总耗时过长而超时。
    fast_mode = source_text_len > FAST_MODE_SOURCE_CHARS
    if chapter_count > 0:
        target_scene_count = chapter_count
    else:
        target_scene_count = (
            FAST_MODE_TARGET_SCENE_COUNT if fast_mode else DEFAULT_TARGET_SCENE_COUNT
        )
    enable_review = source_text_len <= REVIEW_ENABLED_MAX_SOURCE_CHARS
    max_iterations = calculate_max_iterations(target_scene_count, enable_review)

    # 这里把主系统的 AgentState 映射成创作子图自己的状态。
    # 这样做的好处是：
    # 1. ReAct 子图可以独立演进，不污染主状态
    # 2. 创作任务需要的 scene_plan / scene_drafts / review_notes
    #    都可以集中管理
    react_state: NovelScriptState = {
        "request_id": state.get("request_id", ""),
        "session_id": state.get("session_id", "default"),
        "debug": state.get("debug", False),
        "stream_callback": get_stream_callback(),
        "streamed_answer": False,
        "source_text": source_text,
        "task_goal": "把小说片段改写成结构化剧本",
        "script_style": "影视短剧",
        "target_scene_count": target_scene_count,
        "enable_review": enable_review,
        "fast_mode": fast_mode,
        "scene_plan": [],
        "story_facts": {},
        "scene_drafts": [],
        "review_notes": [],
        "pending_rewrite_scene_ids": [],
        "pending_rewrite_reasons": {},
        "scene_rewrite_attempts": {},
        "draft_version": 0,
        "last_reviewed_draft_version": -1,
        "iteration_timings": [],
        "timing_breakdown_ms": {},
        "tool_history": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "max_scene_rewrite_attempts": MAX_SCENE_REWRITE_ATTEMPTS,
        "done": False,
    }

    # 真正的 ReAct 循环发生在子图里：
    # planner -> tool_executor -> planner ... -> finalizer
    result = novel_script_graph.invoke(react_state)
    answer = result.get("final_script", "") or INSUFFICIENT_KNOWLEDGE_ANSWER

    # 把子图内部的中间产物透出到 debug_info，
    # 方便你在 debug 模式下观察“它到底做了几轮、用了哪些动作”。
    novel_debug = {
        "source_text_preview": preview(source_text, 160),
        "source_text_len": source_text_len,
        "chapter_count": chapter_count,
        "llm_profiles": {
            PROFILE_CREATIVE_PLANNER: get_profile_runtime_info(
                PROFILE_CREATIVE_PLANNER
            ),
            PROFILE_CREATIVE_WRITE: get_profile_runtime_info(PROFILE_CREATIVE_WRITE),
            PROFILE_CREATIVE_REVIEW: get_profile_runtime_info(PROFILE_CREATIVE_REVIEW),
        },
        "scene_count": len(result.get("scene_plan", [])),
        "draft_count": len(result.get("scene_drafts", [])),
        "review_count": len(result.get("review_notes", [])),
        "iteration_count": result.get("iteration_count", 0),
        "timing_breakdown_ms": result.get("timing_breakdown_ms", {}),
        "iteration_timings": result.get("iteration_timings", []),
        "tool_history": result.get("tool_history", []),
        "finalizer_strategy": (
            result.get("iteration_timings", [])[-1].get("strategy", "")
            if result.get("iteration_timings")
            else ""
        ),
    }

    next_state: AgentState = {
        "agent_outputs": {ROUTE_NOVEL_SCRIPT_AGENT: answer},
        "answer": answer,
        "debug_info": {ROUTE_NOVEL_SCRIPT_AGENT: novel_debug},
    }
    if result.get("streamed_answer"):
        next_state["streamed_answer"] = True
    log_state = {**state, **next_state}

    log_node(
        ROUTE_NOVEL_SCRIPT_AGENT,
        log_state,
        extra={
            "sceneCount": len(result.get("scene_plan", [])),
            "draftCount": len(result.get("scene_drafts", [])),
            "reviewCount": len(result.get("review_notes", [])),
            "iterationCount": result.get("iteration_count", 0),
            "sourceTextLen": source_text_len,
            "chapterCount": chapter_count,
            "llmProfiles": {
                PROFILE_CREATIVE_PLANNER: get_profile_runtime_info(
                    PROFILE_CREATIVE_PLANNER
                ),
                PROFILE_CREATIVE_WRITE: get_profile_runtime_info(
                    PROFILE_CREATIVE_WRITE
                ),
                PROFILE_CREATIVE_REVIEW: get_profile_runtime_info(
                    PROFILE_CREATIVE_REVIEW
                ),
            },
            "timingBreakdownMs": result.get("timing_breakdown_ms", {}),
            "finalAnswerPreview": preview(answer, 160),
        },
    )
    return next_state

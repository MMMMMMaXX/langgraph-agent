from typing import Literal

from typing_extensions import TypedDict


class NovelScriptState(TypedDict, total=False):
    request_id: str
    session_id: str
    debug: bool
    stream_callback: object
    streamed_answer: bool

    source_text: str
    task_goal: str
    script_style: str
    target_scene_count: int
    enable_review: bool
    fast_mode: bool

    # ReAct 决策相关字段：
    # thought / selected_tool / tool_input / tool_output / observation
    # 共同描述了一轮“思考 -> 动作 -> 观察”的闭环。
    thought: str
    selected_tool: Literal[
        "split_into_scenes",
        "extract_story_facts",
        "write_script_scene",
        "review_script",
        "finalize",
    ]
    tool_input: dict
    tool_output: dict
    observation: str
    tool_history: list[dict]

    # 创作任务的关键中间产物。
    # 这些字段会被多轮循环逐步填充，而不是一次性生成出来。
    scene_plan: list[dict]
    story_facts: dict
    scene_drafts: list[dict]
    review_notes: list[dict]
    pending_rewrite_scene_ids: list[str]
    pending_rewrite_reasons: dict[str, str]
    scene_rewrite_attempts: dict[str, int]
    draft_version: int
    last_reviewed_draft_version: int
    iteration_timings: list[dict]
    timing_breakdown_ms: dict[str, float]

    # 循环控制字段，用来防止 agent 无限制迭代。
    iteration_count: int
    max_iterations: int
    max_scene_rewrite_attempts: int
    done: bool
    final_script: str

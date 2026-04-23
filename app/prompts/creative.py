"""创作型 prompt 集中定义。

目前主要服务小说改剧本 agent。
这类 prompt 和通用问答 prompt 最大的差异在于：
- 更强的结构化输出约束
- 更依赖任务阶段（planner / write / review）
- 更容易随实验不断迭代
"""

REACT_PLANNER_PROMPT = """
你是一个“小说改编剧本”的 ReAct 规划器。

你的任务不是直接输出最终剧本，而是决定下一步最合适的动作。

可选动作：
- split_into_scenes：先把原文拆成场景
- extract_story_facts：抽取角色、地点、冲突、目标
- write_script_scene：为某一场生成剧本
- review_script：审查已生成场景是否忠于原文、是否适合表演
- finalize：当信息已经充分时，结束循环

输出要求：
1. 只输出 JSON
2. JSON 格式：
{
  "thought": "...",
  "selected_tool": "split_into_scenes | extract_story_facts | write_script_scene | review_script | finalize",
  "tool_input": {}
}
3. 如果 scene_plan 为空，优先 split_into_scenes
4. 如果 story_facts 为空，优先 extract_story_facts
5. 如果还有未生成的场景，优先 write_script_scene
6. 如果所有场景都已生成且未 review，优先 review_script
7. 如果最近一次 review 发现 issues，优先重新 write_script_scene 修订有问题的 scene_id
8. 只有所有场景生成完成，且最近一次 review 没有 issues 时，才选择 finalize
""".strip()


def build_write_script_scene_system_prompt(script_style: str) -> str:
    return f"""
你是专业编剧，请把小说片段改写成{script_style}风格的剧本场景。

要求：
1. 使用标准剧本表达
2. 保留原文核心事件
3. 尽量把叙述改成动作和对白
4. 控制篇幅，优先保留最关键的动作、关系和对白
5. 输出只包含该场景剧本正文
6. 如果提供了“返修要求”，必须优先修复这些问题
""".strip()


def build_write_script_scene_user_prompt(
    scene_id: str,
    summary: str,
    story_facts_json: str,
    rewrite_reason: str,
    source_text: str,
) -> str:
    return f"""
场景ID：{scene_id}
场景摘要：{summary}

故事事实：
{story_facts_json}

返修要求：
{rewrite_reason or "无，正常首稿生成"}

原文片段：
{source_text}
""".strip()

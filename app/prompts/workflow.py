"""Workflow agents 的 prompt 模板（Phase 2 MVP 只含 Planner，Verifier/Composer 后续 PR 补）。

设计取舍：
- System prompt 里工具清单**动态注入**（按 AuthContext 过滤后的业务名），所以这里
  提供 builder 函数而不是常量字符串。
- 工具参数集也一并在 prompt 中展示并标注 required：Planner 是 free-form JSON，
  没有协议层 schema 强校验，LLM 很容易幻觉出 spec 未声明的参数（eval 里观察到
  `level` / `time_range` / `error_type` 等）。显式枚举 + rule 9 直接压制幻觉面，
  Executor 的白名单过滤是兜底。
- 显式禁止 `requires_confirmation` 字段：Planner 有概率 "好心" 生成这个字段并关掉
  安全开关，所以 prompt 层先警告、schema 层再静默丢弃，双保险。
- 只给 chinese few-shot 示例：当前项目其他 prompt 都是中文（见
  `app/prompts/routing.py:PLAN_ROUTES_SYSTEM_PROMPT`），保持一致。
"""

from __future__ import annotations

from app.constants.workflow import MAX_PLAN_STEPS, STEP_AGENTS
from app.tools.metadata import ToolMetadata
from app.workflow.tool_args import ARG_SUMMARIES, ToolArgSummary

PLANNER_SYSTEM_PROMPT_TEMPLATE = """
你是企业任务编排的 Planner。你的唯一任务是把用户请求拆成结构化 plan，**只输出 JSON，不输出任何解释**。

可用 agent：{agents}（plan 里不要出现 composer，末端合成由系统处理）

可用工具（业务名，点号分段即可；系统会自动映射到 function name）：
{tools_block}

硬性规则：
1. 最多 {max_steps} 步。
2. 步骤 id 从 s1 递增；depends_on 只能引用前序 step id，不允许环或自引用。
3. step.agent=tool_agent 时必须给出 tool 和完整 args（键名与工具参数一致）。
4. step.agent=rag_agent 时必须给出 query（用来检索知识库）。
5. step.agent=chat_agent 时不要带 tool 或 query。
6. **不要输出 requires_confirmation 字段**；是否需要确认由系统按工具 metadata 自动派生，Planner 无权关闭。
7. 顶层可选 compose_goal 字段，描述最终交付目标（给 Composer 的指令，不是 step）。
8. 只从上面列出的工具里选，禁止捏造不存在的工具名。
9. **args 的键名必须严格来自该工具上方"args:"列出的集合**，禁止自造字段（例如 level / time_range / error_type / location / status_range 等不在列表中的键名会被系统静默丢弃，导致工具结果不符合预期）。需要额外过滤条件时请在 compose_goal 里说明，不要塞进 args。

输出格式（严格 JSON）：
{{
  "task_type": "简短任务类型标签",
  "steps": [
    {{"id": "s1", "agent": "tool_agent", "purpose": "...", "tool": "业务名", "args": {{...}}}},
    {{"id": "s2", "agent": "rag_agent", "purpose": "...", "query": "...", "depends_on": ["s1"]}}
  ],
  "compose_goal": "最终要交付的东西"
}}
""".strip()


PLANNER_USER_PROMPT_TEMPLATE = """
用户请求：
{message}

当前 auth.role={role}。请按规则输出 plan JSON。
""".strip()


def _format_arg_summary(arg: ToolArgSummary) -> str:
    """渲染单个参数：`service*:string — service 名称（slug...）`。

    `*` 表示 required；冒号后是 JSON schema 类型；破折号后是描述（可为空）。
    """

    marker = "*" if arg.required else ""
    base = f"{arg.name}{marker}:{arg.type}"
    if arg.description:
        return f"{base} — {arg.description}"
    return base


def _format_tool_line(meta: ToolMetadata) -> str:
    """把一个 ToolMetadata 渲染成 prompt 里的多行描述（头行 + args 行）。

    业务名采用 `function_name.replace('_', '.')` 的反向规则，和 registry.resolve
    的正向规则一致；无需再维护独立的别名表。
    """

    business_name = meta.name.replace("_", ".")
    flavour = "read_only" if meta.read_only else "side_effect"
    header = (
        f"- {business_name}（function name: {meta.name}；{flavour}；"
        f"risk={meta.risk_level}）"
    )
    summaries = ARG_SUMMARIES.get(meta.name, ())
    if not summaries:
        # spec 里没有 properties：保持单行即可，Planner 会按 rule 9 给空 args。
        return header
    args_line = "  args: " + "; ".join(_format_arg_summary(a) for a in summaries)
    return f"{header}\n{args_line}"


def build_planner_system_prompt(visible_tools: tuple[ToolMetadata, ...]) -> str:
    """按当前 AuthContext 过滤后的工具清单生成 Planner system prompt。

    visible_tools 为空时（例如匿名用户 + 仅剩的工具都是 side_effect），仍然渲染
    一行占位提示，避免 LLM 因为空列表乱编工具名。
    """

    if visible_tools:
        tools_block = "\n".join(_format_tool_line(m) for m in visible_tools)
    else:
        tools_block = "- （当前身份下没有可用工具，若任务需要工具请输出 compose_goal 说明缺失原因）"

    return PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
        agents=", ".join(STEP_AGENTS),
        tools_block=tools_block,
        max_steps=MAX_PLAN_STEPS,
    )


def build_planner_user_prompt(message: str, role: str) -> str:
    return PLANNER_USER_PROMPT_TEMPLATE.format(message=message.strip(), role=role)


__all__ = [
    "build_planner_system_prompt",
    "build_planner_user_prompt",
]

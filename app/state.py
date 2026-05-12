from typing import Annotated, Any, Literal

from typing_extensions import TypedDict

from app.auth.context import AuthContext


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """合并 LangGraph state 中的字典字段。

    用于 agent_outputs/debug_info/node_timings 这类“多节点逐步追加”的字段。
    节点只需要返回本轮新增的 key，LangGraph 会通过 reducer 自动合并，避免
    每个节点手写 dict(state.get(...)) 时遗漏防御性拷贝。
    """

    return {**(left or {}), **(right or {})}


class AgentState(TypedDict, total=False):
    request_id: str
    session_id: str
    debug: bool
    conversation_history_path: str
    stream_callback: Any
    streamed_answer: bool
    messages: list[dict]
    summary: str

    # Phase 1：身份上下文，Supervisor / 各 agent / tool_agent 都按需消费。
    # 非 Optional：API 层会在构造 state 前完成注入（匿名 fallback 或 401），
    # 下游节点默认认为它一定存在；测试路径可直接传匿名 AuthContext。
    auth: AuthContext

    # Phase 1：side_effect 工具返回 need_confirmation 时填充，下一次请求
    # 带 confirmation_token 进来后由 tool_agent 验证并清空。
    # 结构：{tool_name, args, idempotency_key, expires_at, token}
    pending_confirmation: dict

    # Phase 1：本次请求执行过的 side_effect 工具记录，便于 eval / debug
    # 观察抢占、幂等、超时行为。不做跨请求累积，graph 每轮新建。
    tool_executions: list[dict]

    # Phase 1：二次请求携带的 confirmation_token（chat_runner 注入），
    # tool_agent 校验通过后开始执行实际工具。
    confirmation_token: str

    # supervisor 决策
    routes: list[
        Literal[
            "rag_agent",
            "tool_agent",
            "chat_agent",
            "novel_script_agent",
            "workflow_agent",
        ]
    ]

    # 中间结果
    rewritten_query: str
    context: str
    tool_result: str
    agent_outputs: Annotated[dict, merge_dict]

    # Phase 2：Planner 输出的结构化 plan（序列化为 dict，内部结构见
    # `app/workflow/schema.py`）。非 workflow 请求为空 dict。
    plan: dict

    # Phase 2：Plan 的唯一 id（Planner 生成的 uuid4.hex），用于 trace / debug 聚合。
    # 非 workflow 请求为空字符串。
    plan_id: str

    # Phase 2：Workflow Executor 逐 step 写入的执行结果，key=step.id。
    # 每个值形如：{"status": "succeeded"|"failed"|..., "output": str, "tool_executions": [...]}
    # merge_dict 允许 Executor 分批追加，而不是一次性替换整张表。
    step_results: Annotated[dict, merge_dict]

    # Phase 2：Workflow 整体状态（对齐总设 §8.1 枚举），见
    # `app/constants/workflow.py:VALID_WORKFLOW_STATUSES`。非 workflow 请求为空。
    workflow_status: str

    # Phase 2：Verifier 的结构化输出（PR-3 真正填充）；Planner 失败时也写此字段
    # 的 unsupported_claims 方便 Composer 合成拒绝文案。
    verification: dict

    # 新增：vector memory 检索结果
    memory_hits: list[dict]
    node_timings: Annotated[dict, merge_dict]
    debug_info: Annotated[dict, merge_dict]

    # 最终输出
    answer: str

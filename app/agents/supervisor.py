from app.agents.novel_script_agent import looks_like_script_task
from app.agents.rag.multi_hop.gate import should_enter_multi_hop
from app.constants.keywords import (
    FOLLOWUP_QUERY_MAX_CHARS,
    FOLLOWUP_QUERY_PREFIXES,
    FOLLOWUP_QUERY_SUFFIXES,
    KNOWLEDGE_QUERY_KEYWORDS,
    MATH_OPERATOR_KEYWORDS,
    MATH_QUERY_KEYWORDS,
    META_HISTORY_QUERY_KEYWORDS,
    META_HISTORY_QUERY_MAX_CHARS,
    REPEAT_QUERY_KEYWORDS,
    SUMMARY_QUERY_KEYWORDS,
    TICKET_QUERY_KEYWORDS,
    WEATHER_QUERY_KEYWORDS,
    WORKFLOW_QUERY_KEYWORDS,
    contains_any,
)
from app.constants.routes import (
    ROUTE_CHAT_AGENT,
    ROUTE_MULTI_HOP_AGENT,
    ROUTE_NOVEL_SCRIPT_AGENT,
    ROUTE_RAG_AGENT,
    ROUTE_TOOL_AGENT,
    ROUTE_WORKFLOW,
)
from app.constants.workflow import INTENT_WORKFLOW
from app.llm import plan_routes
from app.state import AgentState
from app.utils.logger import log_node, preview


def looks_like_math_query(message: str) -> bool:
    # 只有同时出现数字和数学运算符时，才认为这是一条计算类请求。
    # 这样像 "WAI-ARIA" 这种带连字符的技术名，就不会被误判成减法。
    return any(ch.isdigit() for ch in message) and any(
        op in message for op in MATH_OPERATOR_KEYWORDS
    )


def is_weather_query(message: str) -> bool:
    return contains_any(message, WEATHER_QUERY_KEYWORDS)


def is_knowledge_query(message: str) -> bool:
    # “天气”偏实时工具查询，“气候/是什么/原理”偏知识检索。
    # 这里避免因为出现城市名就把纯天气问题误路由到 RAG。
    if contains_any(message, KNOWLEDGE_QUERY_KEYWORDS):
        return True

    return False


def is_short_followup_query(message: str, messages: list[dict]) -> bool:
    """判断是否是依赖上一轮主题的短追问。

    例如“那上海呢”本身不包含“气候”，但上一轮如果是“北京气候怎么样”，
    RAG 的 rewrite 会补全成“上海气候怎么样？”。这里先做非常窄的规则，
    只识别短句，避免把普通闲聊大面积误路由到 RAG。
    """

    stripped = message.strip()
    previous_user_count = sum(1 for item in messages[:-1] if item.get("role") == "user")
    if previous_user_count == 0 or len(stripped) > FOLLOWUP_QUERY_MAX_CHARS:
        return False

    starts_like_followup = stripped.startswith(FOLLOWUP_QUERY_PREFIXES)
    ends_like_followup = stripped.endswith(FOLLOWUP_QUERY_SUFFIXES)
    return starts_like_followup and ends_like_followup


def is_meta_history_query(message: str) -> bool:
    # 历史/总结类查询通常是短指令。
    # 如果直接对整段长文本做关键词命中，很容易把创作正文误判成“历史查询”。
    if len(message.strip()) > META_HISTORY_QUERY_MAX_CHARS:
        return False

    return contains_any(message, META_HISTORY_QUERY_KEYWORDS)


def looks_like_workflow_task(message: str) -> bool:
    """识别多步 workflow 请求。

    判定口径（命中任一即可）：
    - 显式 workflow 连接词/动作词（`WORKFLOW_QUERY_KEYWORDS`）：用户明确在
      表达"先 X 再 Y"的顺序。
    - 同一句里同时出现"查询类"关键词 + "副作用类"关键词：典型"先查后操作"
      模式（如"查一下北京天气，如果太热就提个工单"）。

    只看本轮消息，不做跨轮累加——避免把"历史提过单"这种上下文误升级成
    workflow。
    """

    if contains_any(message, WORKFLOW_QUERY_KEYWORDS):
        return True

    has_query_intent = contains_any(
        message, (*WEATHER_QUERY_KEYWORDS, *KNOWLEDGE_QUERY_KEYWORDS)
    )
    has_side_effect_intent = contains_any(message, TICKET_QUERY_KEYWORDS)
    return has_query_intent and has_side_effect_intent


def supervisor_node(state: AgentState) -> AgentState:
    message = state["messages"][-1]["content"].strip()

    routes = []
    route_reason = ""
    intent = ""

    # Safety-First Routing（Phase 2）：
    # 二次请求携带 confirmation_token 时短路回 tool_agent，绕开 Planner。
    # 若让 Planner 重新解析 user 文案会产生 args drift（例如第一次
    # "开个标题是改 bug 的工单"、第二次"确认一下刚才的请求"——Planner 会
    # 产出完全不同的 args），而 token 已经冻住了 tool_name + args 的哈希。
    # 交给 tool_agent 的 Scheme A 直接重放 pipeline，幂等闭环才成立。
    confirmation_token = (state.get("confirmation_token") or "").strip()
    if confirmation_token:
        routes = [ROUTE_TOOL_AGENT]
        route_reason = "confirmation_token short-circuit"
        intent = "tool"
        log_node(
            "supervisor",
            state,
            extra={
                "routeReason": route_reason,
                "summaryPreview": preview(state.get("summary", ""), 120),
            },
        )
        return {
            "routes": routes,
            "intent": intent,
            "debug_info": {
                "supervisor": {
                    "route_reason": route_reason,
                    "intent": intent,
                    "routes": routes,
                }
            },
        }

    # 纯 LLM Supervisor 有两个问题：
    # 稳定性不如规则
    # 成本更高

    # 所以更合理的是：
    # 规则优先 + LLM fallback

    # 你在面试里也可以直接这么讲：
    # 在 supervisor 上我采用 hybrid routing：高确定性模式先走规则，模糊查询再交给 LLM planner，提高稳定性和泛化能力的平衡。

    # 先走规则
    # 创作型请求优先级更高。
    # 否则长小说正文里只要碰巧出现“之前/是否”之类字样，
    # 就会被错误路由到 chat_agent。
    if looks_like_script_task(message):
        routes = [ROUTE_NOVEL_SCRIPT_AGENT]
        route_reason = "creative script adaptation query"
    elif is_meta_history_query(message):
        routes = [ROUTE_CHAT_AGENT]
        route_reason = "meta/history query"
    elif contains_any(message, (*SUMMARY_QUERY_KEYWORDS, *REPEAT_QUERY_KEYWORDS)):
        routes = [ROUTE_CHAT_AGENT]
        route_reason = "summary query"
    elif looks_like_workflow_task(message):
        # 多步编排请求：交给 Planner → Executor。放在创作/历史/总结之后
        # 是因为这些是"单一清晰意图"，不该被 workflow 关键词偷走；但要
        # 早于单 agent 的规则路由，避免"查天气再提工单"被误拆成单 tool。
        routes = [ROUTE_WORKFLOW]
        route_reason = "multi-step workflow query"
    else:
        weather_query = is_weather_query(message)
        knowledge_query = is_knowledge_query(message)
        followup_query = is_short_followup_query(message, state["messages"])
        math_query = contains_any(
            message, MATH_QUERY_KEYWORDS
        ) or looks_like_math_query(message)

        if weather_query or math_query:
            routes.append(ROUTE_TOOL_AGENT)

        # Phase 3 multi-hop gate：独立于 KNOWLEDGE_QUERY_KEYWORDS。
        # 很多典型多跳请求（"基于A项目接口文档和B项目部署逻辑写集成测试方案"）
        # 不含"是什么/原理/技术"等知识关键词，但语义上依赖跨文档推理。
        # 所以只要不是 tool 类诉求，就交给 `should_enter_multi_hop` 自己判（它已经
        # 先过 negative gates，再过 positive triggers，不会错拦定义/对比类）。
        if not weather_query and not math_query and should_enter_multi_hop(message):
            routes.append(ROUTE_MULTI_HOP_AGENT)
        elif knowledge_query or (
            followup_query and not weather_query and not math_query
        ):
            # 知识型问题再走 RAG，避免"北京天气怎么样"因为带城市名被误拉到知识库链路。
            routes.append(ROUTE_RAG_AGENT)

        # 规则没兜住，再让 LLM 规划
        if routes:
            route_reason = "rule-based route"
        else:
            routes = plan_routes(message)
            route_reason = "llm fallback"

    # 生成 intent
    if routes == [ROUTE_CHAT_AGENT]:
        intent = "chat"
    elif routes == [ROUTE_NOVEL_SCRIPT_AGENT]:
        intent = "creative"
    elif routes == [ROUTE_WORKFLOW]:
        intent = INTENT_WORKFLOW
    elif ROUTE_MULTI_HOP_AGENT in routes:
        intent = "multi_hop_retrieval"
    elif ROUTE_TOOL_AGENT in routes and ROUTE_RAG_AGENT in routes:
        intent = "hybrid"
    elif ROUTE_TOOL_AGENT in routes:
        intent = "tool"
    elif ROUTE_RAG_AGENT in routes:
        intent = "retrieval"
    else:
        intent = "chat"

    log_node(
        "supervisor",
        state,
        extra={
            "routeReason": route_reason,
            "summaryPreview": preview(state.get("summary", ""), 120),
        },
    )
    return {
        "routes": routes,
        "intent": intent,
        "debug_info": {
            "supervisor": {
                "route_reason": route_reason,
                "intent": intent,
                "routes": routes,
            }
        },
    }

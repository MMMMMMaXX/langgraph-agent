from app.constants.keywords import (
    FOLLOWUP_QUERY_PREFIXES,
    FOLLOWUP_QUERY_SUFFIXES,
    FOLLOWUP_QUERY_MAX_CHARS,
    KNOWLEDGE_QUERY_KEYWORDS,
    MATH_OPERATOR_KEYWORDS,
    MATH_QUERY_KEYWORDS,
    META_HISTORY_QUERY_KEYWORDS,
    META_HISTORY_QUERY_MAX_CHARS,
    REPEAT_QUERY_KEYWORDS,
    SUMMARY_QUERY_KEYWORDS,
    WEATHER_QUERY_KEYWORDS,
    contains_any,
)
from app.constants.routes import (
    ROUTE_CHAT_AGENT,
    ROUTE_NOVEL_SCRIPT_AGENT,
    ROUTE_RAG_AGENT,
    ROUTE_TOOL_AGENT,
)
from app.state import AgentState
from app.llm import plan_routes
from app.agents.novel_script_agent import looks_like_script_task
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


def supervisor_node(state: AgentState) -> AgentState:
    message = state["messages"][-1]["content"].strip()

    routes = []
    route_reason = ""
    intent = ""

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
    else:
        weather_query = is_weather_query(message)
        knowledge_query = is_knowledge_query(message)
        followup_query = is_short_followup_query(message, state["messages"])
        math_query = contains_any(message, MATH_QUERY_KEYWORDS) or looks_like_math_query(
            message
        )

        if weather_query or math_query:
            routes.append(ROUTE_TOOL_AGENT)

        # 知识型问题再走 RAG，避免“北京天气怎么样”因为带城市名被误拉到知识库链路。
        if knowledge_query or (followup_query and not weather_query and not math_query):
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

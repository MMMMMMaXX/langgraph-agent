from dataclasses import asdict, dataclass

from app.constants.keywords import VECTOR_STORE_BLOCK_KEYWORDS, contains_any
from app.constants.policies import (
    SKIP_REASON_BAD_ANSWER,
    SKIP_REASON_CREATIVE_OUTPUT,
    SKIP_REASON_EMPTY_ANSWER,
    SKIP_REASON_META_QUERY,
    SKIP_REASON_RAG_DOC_HIT,
    SKIP_REASON_TOO_SHORT,
    SKIP_REASON_TOOL_REQUEST,
)
from app.constants.routes import (
    ROUTE_NOVEL_SCRIPT_AGENT,
    ROUTE_RAG_AGENT,
    ROUTE_TOOL_AGENT,
)
from app.memory.vector_memory import DEFAULT_MEMORY_CONFIDENCE
from app.state import AgentState
from app.utils.memory_key import (
    MEMORY_TYPE_FACT,
    build_memory_key,
    classify_memory_type,
)
from app.utils.tags import extract_tags


@dataclass(frozen=True)
class MemoryWriteDecision:
    should_write: bool
    skip_reason: str = ""
    memory_type: str = MEMORY_TYPE_FACT
    tags: tuple[str, ...] = ()
    memory_key: str = ""
    confidence: float = DEFAULT_MEMORY_CONFIDENCE
    source_route: str = ""

    def to_debug_dict(self) -> dict:
        data = asdict(self)
        data["tags"] = list(self.tags)
        return data


def _joined_routes(state: AgentState) -> str:
    return ",".join(state.get("routes", []))


def _skip_reason_from_route(state: AgentState) -> str:
    routes = state.get("routes", [])
    tool_debug = state.get("debug_info", {}).get(ROUTE_TOOL_AGENT, {})
    tool_calls = tool_debug.get("tool_calls", [])
    rag_debug = state.get("debug_info", {}).get(ROUTE_RAG_AGENT, {})

    # 工具型结果大多是短期信息，如计算和天气，默认不进入长期语义记忆。
    if routes == [ROUTE_TOOL_AGENT] and len(tool_calls) >= 1:
        return SKIP_REASON_TOOL_REQUEST

    # 创作输出通常很长，也更适合放在文件/产物里，不适合整段写入 Chroma。
    if routes == [ROUTE_NOVEL_SCRIPT_AGENT]:
        return SKIP_REASON_CREATIVE_OUTPUT

    # RAG 文档命中答案的事实来源是 docs，长期复用时应重新检索 docs，
    # 不把模型生成答案反写到 semantic memory，避免 docs 与 memory 形成重复事实源。
    if routes == [ROUTE_RAG_AGENT] and rag_debug.get("doc_used"):
        return SKIP_REASON_RAG_DOC_HIT

    return ""


def _skip_reason_from_content(answer: str, rewritten_query: str) -> str:
    if not answer:
        return SKIP_REASON_EMPTY_ANSWER
    if answer == "资料不足" or "无法处理" in answer:
        return SKIP_REASON_BAD_ANSWER
    if len(answer) < 8:
        return SKIP_REASON_TOO_SHORT
    if contains_any(rewritten_query, VECTOR_STORE_BLOCK_KEYWORDS):
        return SKIP_REASON_META_QUERY
    return ""


def decide_memory_write(
    *,
    state: AgentState,
    user_message: str,
    rewritten_query: str,
    answer: str,
) -> MemoryWriteDecision:
    """统一判断当前轮是否应该写入 Chroma semantic memory。

    这个函数只负责“决策”，不做 embedding、不写 Chroma。
    memory_node 根据返回值执行写入，并把完整 decision 暴露到 debug。
    """

    tags = tuple(extract_tags(rewritten_query or user_message or answer))
    memory_type = classify_memory_type(rewritten_query)
    memory_key = build_memory_key(rewritten_query or user_message or answer, list(tags))
    source_route = _joined_routes(state)

    route_skip_reason = _skip_reason_from_route(state)
    if route_skip_reason:
        return MemoryWriteDecision(
            should_write=False,
            skip_reason=route_skip_reason,
            memory_type=memory_type,
            tags=tags,
            memory_key=memory_key,
            source_route=source_route,
        )

    content_skip_reason = _skip_reason_from_content(answer, rewritten_query)
    if content_skip_reason:
        return MemoryWriteDecision(
            should_write=False,
            skip_reason=content_skip_reason,
            memory_type=memory_type,
            tags=tags,
            memory_key=memory_key,
            source_route=source_route,
        )

    return MemoryWriteDecision(
        should_write=True,
        memory_type=memory_type,
        tags=tags,
        memory_key=memory_key,
        confidence=DEFAULT_MEMORY_CONFIDENCE,
        source_route=source_route,
    )

from app.agents.chat.constants import (
    CHAT_MEMORY_SEARCH_TOP_K,
    CHAT_QA_RERANK_TOP_K,
    CHAT_RECALL_RERANK_TOP_K,
    LOW_VALUE_MEMORY_BLOCK_KEYWORDS,
    LOW_VALUE_MEMORY_MIN_CONTENT_CHARS,
    TASK_RECALL,
    TASK_SUMMARY,
)
from app.agents.chat.intent import classify_chat_task
from app.agents.chat.policies import choose_memory_lookup_policy
from app.constants.policies import MEMORY_POLICY_WORKING_ONLY
from app.constants.tags import CITY_TAGS, TOPIC_TAGS
from app.memory.vector_memory import search_memory
from app.retrieval.reranker import rerank
from app.utils.errors import build_error_info
from app.utils.memory_key import MEMORY_TYPE_FACT, dedupe_memory_hits


def build_memory_facts(memory_hits: list[dict]) -> dict:
    """从 memory hits 中抽取结构化事实，供回答 prompt 使用。"""

    cities = []
    topics = []
    memory_types = []

    for m in memory_hits:
        tags = m.get("tags", [])
        for t in tags:
            if t in CITY_TAGS and t not in cities:
                cities.append(t)
            if t in TOPIC_TAGS and t not in topics:
                topics.append(t)

        mt = m.get("memory_type")
        if mt and mt not in memory_types:
            memory_types.append(mt)

    return {
        "cities": cities,
        "topics": topics,
        "memory_types": memory_types,
    }


def build_structured_facts_text(facts: dict) -> str:
    """把结构化事实渲染成 prompt 文本。"""

    cities = "、".join(facts.get("cities", [])) or "无"
    topics = "、".join(facts.get("topics", [])) or "无"
    memory_types = "、".join(facts.get("memory_types", [])) or "无"

    return f"""结构化事实：
- 涉及城市：{cities}
- 涉及主题：{topics}
- 记忆类型：{memory_types}
""".strip()


def filter_low_value_memory(memory_hits: list[dict]) -> list[dict]:
    """过滤不适合进入回答上下文的低价值 memory。"""

    result = []

    for m in memory_hits:
        content = m.get("content", "")

        # 没有城市也没有主题价值的低质量条目。
        if any(keyword in content for keyword in LOW_VALUE_MEMORY_BLOCK_KEYWORDS):
            continue

        # 太短且没结构信息。
        if len(content.strip()) < LOW_VALUE_MEMORY_MIN_CONTENT_CHARS:
            continue

        result.append(m)

    return result


def prepare_memory_hits(
    message: str,
    session_id: str,
) -> tuple[list[dict], str, str, list[dict], list[str]]:
    """准备 chat agent 使用的 Chroma memory 命中结果。

    返回值保持和旧 `chat_agent.py` 一致，降低拆分风险：
    - memory_hits：最终消费的 memory
    - task：chat task 类型
    - lookup_policy：memory 查询策略
    - memory_before_rerank：rerank 前候选，供 debug 使用
    - errors：检索或 rerank 中产生的错误信息
    """

    task = classify_chat_task(message)
    lookup_policy = choose_memory_lookup_policy(message, task)
    errors: list[str] = []

    try:
        if lookup_policy == MEMORY_POLICY_WORKING_ONLY:
            # 当前请求只允许使用 Working Memory，长期 Chroma memory 不参与，
            # 这样 debug 里的 memory hits 也不会出现旧历史噪音。
            memory_hits = []
        else:
            memory_hits = search_memory(
                message,
                top_k=CHAT_MEMORY_SEARCH_TOP_K,
                session_id=session_id,
            )
    except Exception as exc:
        memory_hits = []
        errors.append(
            build_error_info(
                exc,
                stage="prepare_memory_hits",
                source="memory",
                preferred_code="retrieval_error",
            )
        )
    memory_hits = dedupe_memory_hits(memory_hits)
    memory_hits = filter_low_value_memory(memory_hits)
    memory_before_rerank = memory_hits[:]

    # 回忆类问题目前只消费事实型长期记忆，偏好/纠错/任务状态后续单独设计入口。
    if task == TASK_RECALL:
        # 回忆类问题更看重事实轮次 + 时间。
        memory_hits = [
            m for m in memory_hits if m.get("memory_type") == MEMORY_TYPE_FACT
        ]
        memory_hits = sorted(
            memory_hits,
            key=lambda x: (x.get("timestamp", 0), x.get("score", 0)),
            reverse=True,
        )
        memory_before_rerank = memory_hits[:]
        memory_hits = rerank(message, memory_hits, top_k=CHAT_RECALL_RERANK_TOP_K)

    elif task == TASK_SUMMARY:
        # summary 已统一走 Working Memory + 非向量化会话流水；
        # 这里保持空列表，避免把 Chroma 语义检索结果混进顺序型总结。
        memory_hits = []
        memory_before_rerank = memory_hits[:]

    else:
        # 普通问答类：语义相关优先。
        memory_hits = sorted(
            memory_hits,
            key=lambda x: x.get("score", 0),
            reverse=True,
        )
        memory_before_rerank = memory_hits[:]
        memory_hits = rerank(message, memory_hits, top_k=CHAT_QA_RERANK_TOP_K)

    return memory_hits, task, lookup_policy, memory_before_rerank, errors

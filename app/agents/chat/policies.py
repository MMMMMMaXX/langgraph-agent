from app.agents.chat.intent import TASK_SUMMARY
from app.constants.policies import (
    MEMORY_POLICY_SEMANTIC_LONG_TERM,
    MEMORY_POLICY_WORKING_ONLY,
)


def choose_memory_lookup_policy(message: str, task: str) -> str:
    """决定当前 chat 请求应该如何使用长期 memory。

    Working Memory、长期 Chroma memory、summary 三者语义不同：
    - “刚才/刚刚总结”优先看 Working Memory，再回退到非向量化会话流水
    - “最近所有/回顾/列出”走非向量化会话流水，不再从 Chroma memory 里捞历史
    - 普通回忆/问答才走语义检索

    这个策略函数集中表达边界，避免在多个分支里散落魔法判断。
    """

    if task == TASK_SUMMARY:
        return MEMORY_POLICY_WORKING_ONLY
    return MEMORY_POLICY_SEMANTIC_LONG_TERM

"""兼容汇总层。

现在 prompts 已经按领域拆到多个文件：
- chat.py
- routing.py
- rag.py
- tooling.py
- merge.py
- story.py
- creative.py

保留这个模块的原因：
1. 避免旧 import 立即失效
2. 让迁移可以分阶段进行
3. 仍然支持 `from app.prompts.core import ...` 的旧代码
"""

from app.prompts.chat import (
    CHAT_SUMMARY_SYSTEM_PROMPT,
    SUMMARIZE_MESSAGES_SYSTEM_PROMPT,
    build_chat_qa_system_prompt,
    build_summarize_messages_user_prompt,
)
from app.prompts.merge import MERGE_SYSTEM_PROMPT, build_merge_user_prompt
from app.prompts.rag import (
    RAG_MEMORY_ANSWER_SYSTEM_PROMPT,
    build_rag_doc_answer_system_prompt,
    build_rag_doc_answer_user_prompt,
    build_rag_memory_answer_user_prompt,
)
from app.prompts.routing import (
    PLAN_ROUTES_SYSTEM_PROMPT,
    REWRITE_QUERY_SYSTEM_PROMPT,
    build_route_planning_user_prompt,
)
from app.prompts.story import (
    EXTRACT_STORY_FACTS_SYSTEM_PROMPT,
    REVIEW_SCRIPT_SYSTEM_PROMPT,
)
from app.prompts.tooling import TOOL_AGENT_SYSTEM_PROMPT, build_rerank_prompt


"""LLM 包入口。

重新导出所有 public API，保持 `from app.llm import chat, embed_text, ...` 旧用法可用。
内部结构：
- _helpers.py  : env 解析工具
- providers.py : Provider / ModelProfile + 注册表 + 解析器
- retry.py     : LLMCallError、超时、重试
- caching.py   : embedding 请求级缓存 + OpenAI 客户端进程级缓存
- chat.py      : chat/chat_with_tools + rewrite/summarize/plan_routes
- embedding.py : embed_text / embed_texts

分层依赖（避免循环）：
  _helpers -> retry -> providers -> caching -> {chat, embedding}
"""

# 环境加载必须在构建注册表之前完成。
from app.env import load_project_env

load_project_env()

# --- Public API re-exports --------------------------------------------------
# 顺序按使用频率，便于快速查找。
from app.llm.caching import (  # noqa: E402
    get_embedding_cache_stats,
    reset_embedding_cache,
)
from app.llm.chat import (  # noqa: E402
    chat,
    chat_with_tools,
    plan_routes,
    rewrite_query,
    summarize_messages,
)
from app.llm.embedding import embed_text, embed_texts  # noqa: E402
from app.llm.providers import (  # noqa: E402
    ModelProfile,
    ProviderConfig,
    get_profile_runtime_info,
)
from app.llm.retry import LLMCallError  # noqa: E402

__all__ = [
    # Errors
    "LLMCallError",
    # Providers / profiles
    "ModelProfile",
    "ProviderConfig",
    "get_profile_runtime_info",
    # Chat
    "chat",
    "chat_with_tools",
    "plan_routes",
    "rewrite_query",
    "summarize_messages",
    # Embedding
    "embed_text",
    "embed_texts",
    # Cache control
    "get_embedding_cache_stats",
    "reset_embedding_cache",
]

"""模型 provider、profile 与相关环境变量名。"""

# Provider 名：DeepSeek 兼容 OpenAI SDK，当前作为多数 chat profile 默认厂商。
PROVIDER_DEEPSEEK = "deepseek"

# Provider 名：Embedding 独立配置，默认用于向量化。
PROVIDER_EMBEDDING = "embedding"

# Provider 名：GLM，当前适合轻量审查和判断类任务。
PROVIDER_GLM = "glm"

# Provider 名：OpenAI，作为可选接入。
PROVIDER_OPENAI = "openai"

# 普通聊天 profile：大多数问答、RAG 答案生成默认使用。
PROFILE_DEFAULT_CHAT = "default_chat"

# 创作审查 profile：用于 novel_script review_script。
PROFILE_CREATIVE_REVIEW = "creative_review"

# 创作写作 profile：用于 novel_script write_script_scene。
PROFILE_CREATIVE_WRITE = "creative_write"

# 创作规划 profile：用于 novel_script planner。
PROFILE_CREATIVE_PLANNER = "creative_planner"

# 工具调用 profile：用于 function calling 工具选择与整合。
PROFILE_TOOL_CHAT = "tool_chat"

# 路由规划 profile：用于 supervisor LLM fallback。
PROFILE_ROUTING = "routing"

# 摘要 profile：用于 conversation summary refresh。
PROFILE_SUMMARY = "summary"

# 查询改写 profile：用于 RAG follow-up rewrite。
PROFILE_REWRITE = "rewrite"

# 默认 embedding profile：批量或通用 embedding 默认使用。
PROFILE_DEFAULT_EMBEDDING = "default_embedding"

# 文档索引 embedding profile。
PROFILE_DOC_EMBEDDING = "doc_embedding"

# 记忆写入 embedding profile。
PROFILE_MEMORY_EMBEDDING = "memory_embedding"

# 查询 embedding profile，doc search / memory search 共用。
PROFILE_QUERY_EMBEDDING = "query_embedding"

# 查询分类 profile：低置信度时的 LLM 二裁，轻量模型即可。
PROFILE_CLASSIFY = "classify"

# Provider 选择环境变量：默认 chat provider。
ENV_DEFAULT_CHAT_PROVIDER = "DEFAULT_CHAT_PROVIDER"

# DeepSeek provider 配置环境变量，兼容旧的 API_KEY / BASE_URL / MODEL。
ENV_DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"
ENV_DEEPSEEK_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_DEEPSEEK_MODEL = "DEEPSEEK_MODEL"
ENV_LEGACY_API_KEY = "API_KEY"
ENV_LEGACY_BASE_URL = "BASE_URL"
ENV_LEGACY_MODEL = "MODEL"

# OpenAI provider 配置环境变量。
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
ENV_OPENAI_MODEL = "OPENAI_MODEL"

# GLM provider 配置环境变量。
ENV_GLM_API_KEY = "GLM_API_KEY"
ENV_GLM_BASE_URL = "GLM_BASE_URL"
ENV_GLM_MODEL = "GLM_MODEL"

# Embedding provider 配置环境变量。
ENV_EMBEDDING_API_KEY = "EMBEDDING_API_KEY"
ENV_EMBEDDING_BASE_URL = "EMBEDDING_BASE_URL"

# Provider 选择环境变量：创作审查 provider。
ENV_CREATIVE_REVIEW_PROVIDER = "CREATIVE_REVIEW_PROVIDER"

# Provider 选择环境变量：创作写作 provider。
ENV_CREATIVE_WRITE_PROVIDER = "CREATIVE_WRITE_PROVIDER"

# Provider 选择环境变量：创作规划 provider。
ENV_CREATIVE_PLANNER_PROVIDER = "CREATIVE_PLANNER_PROVIDER"

# Provider 选择环境变量：默认 embedding provider。
ENV_EMBEDDING_PROVIDER = "EMBEDDING_PROVIDER"

# Embedding 模型环境变量。
ENV_EMBEDDING_MODEL = "EMBEDDING_MODEL"

# 请求级 embedding 缓存容量环境变量。
ENV_EMBEDDING_CACHE_MAX_ITEMS = "EMBEDDING_CACHE_MAX_ITEMS"

# 文档 embedding provider / model 环境变量。
ENV_DOC_EMBEDDING_PROVIDER = "DOC_EMBEDDING_PROVIDER"
ENV_DOC_EMBEDDING_MODEL = "DOC_EMBEDDING_MODEL"

# 记忆 embedding provider / model 环境变量。
ENV_MEMORY_EMBEDDING_PROVIDER = "MEMORY_EMBEDDING_PROVIDER"
ENV_MEMORY_EMBEDDING_MODEL = "MEMORY_EMBEDDING_MODEL"

# 查询 embedding provider / model 环境变量。
ENV_QUERY_EMBEDDING_PROVIDER = "QUERY_EMBEDDING_PROVIDER"
ENV_QUERY_EMBEDDING_MODEL = "QUERY_EMBEDDING_MODEL"

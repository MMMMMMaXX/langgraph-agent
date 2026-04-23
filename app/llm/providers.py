"""Provider 与 Profile 的数据模型、注册表及解析逻辑。

概念拆分：
- Provider：基础设施层（DeepSeek / GLM / OpenAI / Embedding），描述"用哪家 API"
- Profile ：业务语义层（default_chat / creative_review / ...），描述"这一跳是什么任务"

这一层只做静态配置与解析，不发任何网络请求。网络请求由 `chat.py` / `embedding.py`
在拿到解析结果后发起。
"""

from dataclasses import dataclass

from app.constants.model_profiles import (
    ENV_CREATIVE_PLANNER_PROVIDER,
    ENV_CREATIVE_REVIEW_PROVIDER,
    ENV_CREATIVE_WRITE_PROVIDER,
    ENV_DEFAULT_CHAT_PROVIDER,
    ENV_DEEPSEEK_API_KEY,
    ENV_DEEPSEEK_BASE_URL,
    ENV_DEEPSEEK_MODEL,
    ENV_DOC_EMBEDDING_MODEL,
    ENV_DOC_EMBEDDING_PROVIDER,
    ENV_EMBEDDING_API_KEY,
    ENV_EMBEDDING_BASE_URL,
    ENV_EMBEDDING_MODEL,
    ENV_EMBEDDING_PROVIDER,
    ENV_GLM_API_KEY,
    ENV_GLM_BASE_URL,
    ENV_GLM_MODEL,
    ENV_LEGACY_API_KEY,
    ENV_LEGACY_BASE_URL,
    ENV_LEGACY_MODEL,
    ENV_MEMORY_EMBEDDING_MODEL,
    ENV_MEMORY_EMBEDDING_PROVIDER,
    ENV_OPENAI_API_KEY,
    ENV_OPENAI_BASE_URL,
    ENV_OPENAI_MODEL,
    ENV_QUERY_EMBEDDING_MODEL,
    ENV_QUERY_EMBEDDING_PROVIDER,
    PROFILE_CREATIVE_PLANNER,
    PROFILE_CREATIVE_REVIEW,
    PROFILE_CREATIVE_WRITE,
    PROFILE_DEFAULT_CHAT,
    PROFILE_DEFAULT_EMBEDDING,
    PROFILE_DOC_EMBEDDING,
    PROFILE_MEMORY_EMBEDDING,
    PROFILE_QUERY_EMBEDDING,
    PROFILE_REWRITE,
    PROFILE_ROUTING,
    PROFILE_SUMMARY,
    PROFILE_TOOL_CHAT,
    PROVIDER_DEEPSEEK,
    PROVIDER_EMBEDDING,
    PROVIDER_GLM,
    PROVIDER_OPENAI,
)

from app.llm._helpers import _env
from app.llm.retry import LLMCallError


@dataclass(frozen=True)
class ProviderConfig:
    """描述一个具体 provider 的接入信息。

    当前先只抽象 OpenAI-compatible 这一类 provider：
    - DeepSeek
    - GLM
    - OpenAI

    它们的差异主要体现在：
    - api_key
    - base_url
    - 默认 model
    """

    name: str
    api_key: str | None
    base_url: str | None
    default_model: str | None


@dataclass(frozen=True)
class ModelProfile:
    """描述"某类任务默认应该使用哪套模型配置"。

    profile 是业务语义：
    - default_chat
    - creative_review
    - creative_write

    provider 是基础设施语义：
    - deepseek
    - glm
    - openai

    这样 agent 只关心"当前任务属于哪个 profile"，不直接依赖厂商名字。
    """

    name: str
    provider: str
    model: str | None = None


def _build_provider_configs() -> dict[str, ProviderConfig]:
    """集中声明当前项目可用的 provider。

    约定：
    - DeepSeek 仍然兼容现有 `API_KEY / BASE_URL / MODEL`
    - GLM / OpenAI 使用各自独立环境变量
    - 如果后续接入更多 provider，只需要在这里扩展
    """

    return {
        PROVIDER_DEEPSEEK: ProviderConfig(
            name=PROVIDER_DEEPSEEK,
            api_key=_env(ENV_DEEPSEEK_API_KEY, ENV_LEGACY_API_KEY)
            or _env(ENV_OPENAI_API_KEY),
            base_url=_env(ENV_DEEPSEEK_BASE_URL, ENV_LEGACY_BASE_URL),
            default_model=_env(ENV_DEEPSEEK_MODEL, ENV_LEGACY_MODEL),
        ),
        PROVIDER_EMBEDDING: ProviderConfig(
            name=PROVIDER_EMBEDDING,
            api_key=_env(ENV_EMBEDDING_API_KEY) or _env(ENV_OPENAI_API_KEY),
            base_url=_env(ENV_EMBEDDING_BASE_URL),
            default_model=_env(ENV_EMBEDDING_MODEL),
        ),
        PROVIDER_GLM: ProviderConfig(
            name=PROVIDER_GLM,
            api_key=_env(ENV_GLM_API_KEY),
            base_url=_env(ENV_GLM_BASE_URL),
            default_model=_env(ENV_GLM_MODEL),
        ),
        PROVIDER_OPENAI: ProviderConfig(
            name=PROVIDER_OPENAI,
            api_key=_env(ENV_OPENAI_API_KEY),
            base_url=_env(ENV_OPENAI_BASE_URL),
            default_model=_env(ENV_OPENAI_MODEL),
        ),
    }


def _build_profile_registry() -> dict[str, ModelProfile]:
    """集中声明当前项目的任务 profile。

    第一版先把 profile 数量控制在项目真正需要的范围内：
    - 大多数任务默认走 `default_chat`
    - creative_review 默认切给 GLM
    - 其他 creative / rag / tool 仍维持默认 DeepSeek 路径
    """

    default_provider = (_env(ENV_DEFAULT_CHAT_PROVIDER) or PROVIDER_DEEPSEEK).lower()
    creative_review_provider = (
        _env(ENV_CREATIVE_REVIEW_PROVIDER) or PROVIDER_GLM
    ).lower()
    creative_write_provider = (
        _env(ENV_CREATIVE_WRITE_PROVIDER) or default_provider
    ).lower()
    creative_planner_provider = (
        _env(ENV_CREATIVE_PLANNER_PROVIDER) or default_provider
    ).lower()

    return {
        PROFILE_DEFAULT_CHAT: ModelProfile(PROFILE_DEFAULT_CHAT, default_provider),
        PROFILE_CREATIVE_REVIEW: ModelProfile(
            PROFILE_CREATIVE_REVIEW, creative_review_provider
        ),
        PROFILE_CREATIVE_WRITE: ModelProfile(
            PROFILE_CREATIVE_WRITE, creative_write_provider
        ),
        PROFILE_CREATIVE_PLANNER: ModelProfile(
            PROFILE_CREATIVE_PLANNER, creative_planner_provider
        ),
        PROFILE_TOOL_CHAT: ModelProfile(PROFILE_TOOL_CHAT, default_provider),
        PROFILE_ROUTING: ModelProfile(PROFILE_ROUTING, default_provider),
        PROFILE_SUMMARY: ModelProfile(PROFILE_SUMMARY, default_provider),
        PROFILE_REWRITE: ModelProfile(PROFILE_REWRITE, default_provider),
    }


def _build_embedding_profile_registry() -> dict[str, ModelProfile]:
    """集中声明 embedding 任务的 profile。

    这里和 chat profile 分开维护，原因是 embedding 更偏基础设施能力：
    - 文档建索引
    - 记忆入库
    - 检索查询

    它们虽然也依赖 provider / model，但不适合混进 creative_write 这类聊天任务语义。
    """

    default_provider = (_env(ENV_EMBEDDING_PROVIDER) or PROVIDER_EMBEDDING).lower()
    return {
        PROFILE_DEFAULT_EMBEDDING: ModelProfile(
            PROFILE_DEFAULT_EMBEDDING,
            default_provider,
            model=_env(ENV_EMBEDDING_MODEL),
        ),
        PROFILE_DOC_EMBEDDING: ModelProfile(
            PROFILE_DOC_EMBEDDING,
            (_env(ENV_DOC_EMBEDDING_PROVIDER) or default_provider).lower(),
            model=_env(ENV_DOC_EMBEDDING_MODEL, ENV_EMBEDDING_MODEL),
        ),
        PROFILE_MEMORY_EMBEDDING: ModelProfile(
            PROFILE_MEMORY_EMBEDDING,
            (_env(ENV_MEMORY_EMBEDDING_PROVIDER) or default_provider).lower(),
            model=_env(ENV_MEMORY_EMBEDDING_MODEL, ENV_EMBEDDING_MODEL),
        ),
        PROFILE_QUERY_EMBEDDING: ModelProfile(
            PROFILE_QUERY_EMBEDDING,
            (_env(ENV_QUERY_EMBEDDING_PROVIDER) or default_provider).lower(),
            model=_env(ENV_QUERY_EMBEDDING_MODEL, ENV_EMBEDDING_MODEL),
        ),
    }


# 模块级注册表：import 时一次性构建，后续只读（进程内不变）。
# 注意：env 在 app/env.py 中于进程启动时加载完成，才能正确 build。
PROVIDER_CONFIGS: dict[str, ProviderConfig] = _build_provider_configs()
CHAT_PROFILE_REGISTRY: dict[str, ModelProfile] = _build_profile_registry()
EMBEDDING_PROFILE_REGISTRY: dict[str, ModelProfile] = _build_embedding_profile_registry()


def _resolve_profile(profile: str | None, kind: str = "chat") -> ModelProfile:
    if kind == "embedding":
        registry = EMBEDDING_PROFILE_REGISTRY
        default_name = PROFILE_DEFAULT_EMBEDDING
    else:
        registry = CHAT_PROFILE_REGISTRY
        default_name = PROFILE_DEFAULT_CHAT

    profile_name = (profile or default_name).strip() or default_name
    return registry.get(profile_name, registry[default_name])


def _resolve_provider(profile: ModelProfile) -> ProviderConfig:
    provider = PROVIDER_CONFIGS.get(profile.provider)
    if provider is not None:
        return provider
    return PROVIDER_CONFIGS[PROVIDER_DEEPSEEK]


def _resolve_model(profile: ModelProfile, provider: ProviderConfig) -> str:
    model = (profile.model or provider.default_model or "").strip()
    if model:
        return model
    raise LLMCallError(
        code="llm_model_missing",
        message=f"profile={profile.name} 未配置可用模型。",
        profile=profile.name,
        provider=provider.name,
        model="",
    )


def get_profile_runtime_info(
    profile: str | None = None,
    kind: str = "chat",
) -> dict[str, str]:
    """返回某个 profile 最终解析到的 provider / model 信息。

    这个方法不发请求，只做静态解析，适合：
    - debug_info 展示
    - timing 日志埋点
    - 在 agent 层快速确认"当前这一跳到底走的是哪家模型"
    """

    profile_config = _resolve_profile(profile, kind=kind)
    provider_config = _resolve_provider(profile_config)
    model = (profile_config.model or provider_config.default_model or "").strip()
    return {
        "kind": kind,
        "profile": profile_config.name,
        "provider": provider_config.name,
        "model": model,
    }

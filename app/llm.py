import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass

from openai import OpenAI

from app.constants.routes import (
    ROUTE_CHAT_AGENT,
    ROUTE_RAG_AGENT,
    ROUTE_TOOL_AGENT,
    ROUTE_NOVEL_SCRIPT_AGENT,
)
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
    ENV_EMBEDDING_CACHE_MAX_ITEMS,
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
from app.env import load_project_env
from app.prompts.chat import (
    SUMMARIZE_MESSAGES_SYSTEM_PROMPT,
    build_summarize_messages_user_prompt,
)
from app.prompts.routing import (
    PLAN_ROUTES_SYSTEM_PROMPT,
    REWRITE_QUERY_SYSTEM_PROMPT,
    build_route_planning_user_prompt,
)
from app.tracing import add_current_run_metadata, build_model_trace_metadata
from app.tracing import next_model_call_index
from app.utils.errors import classify_exception, format_exception_message
from app.utils.logger import log_warning

load_project_env()


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
    """描述“某类任务默认应该使用哪套模型配置”。

    profile 是业务语义：
    - default_chat
    - creative_review
    - creative_write

    provider 是基础设施语义：
    - deepseek
    - glm
    - openai

    这样 agent 只关心“当前任务属于哪个 profile”，不直接依赖厂商名字。
    """

    name: str
    provider: str
    model: str | None = None


def _env(name: str, fallback: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is not None and value.strip() != "":
        return value.strip()
    if fallback is None:
        return None
    fallback_value = os.getenv(fallback)
    if fallback_value is not None and fallback_value.strip() != "":
        return fallback_value.strip()
    return None


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_embedding_cache() -> dict[tuple[str, str, str, str], list[float]]:
    cache = _EMBEDDING_CACHE.get()
    if cache is None:
        cache = {}
        _EMBEDDING_CACHE.set(cache)
    return cache


def _get_embedding_cache_stats() -> dict[str, int]:
    stats = _EMBEDDING_CACHE_STATS.get()
    if stats is None:
        stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
        }
        _EMBEDDING_CACHE_STATS.set(stats)
    return stats


def _get_embedding_cache_max_items() -> int:
    return max(
        0,
        _env_int(ENV_EMBEDDING_CACHE_MAX_ITEMS, DEFAULT_EMBEDDING_CACHE_MAX_ITEMS),
    )


def reset_embedding_cache() -> None:
    """重置请求级 embedding 缓存。

    这个缓存只服务当前 graph/request：
    - 避免同一轮里 doc search 和 memory search 对同一个 query 重复请求 API
    - 不跨请求复用，避免用户隔离、模型切换和长期内存占用问题
    """

    _EMBEDDING_CACHE.set({})
    _EMBEDDING_CACHE_STATS.set(
        {
            "hits": 0,
            "misses": 0,
            "stores": 0,
        }
    )


def get_embedding_cache_stats() -> dict:
    """返回当前请求级 embedding 缓存的调试信息。"""

    cache = _get_embedding_cache()
    stats = dict(_get_embedding_cache_stats())
    stats["size"] = len(cache)
    stats["max_items"] = _get_embedding_cache_max_items()
    return stats


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


PROVIDER_CONFIGS = _build_provider_configs()
CHAT_PROFILE_REGISTRY = _build_profile_registry()
EMBEDDING_PROFILE_REGISTRY = _build_embedding_profile_registry()
_CLIENT_CACHE: dict[str, OpenAI] = {}
# 客户端缓存锁：防止高并发下多个线程同时 miss 后重复创建 OpenAI client，
# 造成连接池浪费。锁只保护字典读写本身，不覆盖任何业务调用。
_CLIENT_CACHE_LOCK = threading.Lock()

DEFAULT_CHAT_TIMEOUT_SECONDS = 45.0
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 30.0
DEFAULT_LLM_MAX_RETRIES = 2
DEFAULT_LLM_RETRY_BACKOFF_SECONDS = 0.5
# 单次请求内最多缓存多少条 embedding，避免异常长请求导致内存无限增长。
DEFAULT_EMBEDDING_CACHE_MAX_ITEMS = 128
NON_RETRYABLE_ERROR_CODES = {"auth_error", "bad_request"}

_EMBEDDING_CACHE: ContextVar[dict[tuple[str, str, str, str], list[float]] | None] = (
    ContextVar("embedding_cache", default=None)
)
_EMBEDDING_CACHE_STATS: ContextVar[dict[str, int] | None] = ContextVar(
    "embedding_cache_stats",
    default=None,
)


class LLMCallError(RuntimeError):
    """统一封装 LLM SDK 调用失败。

    目的不是隐藏原始异常，而是把上游可能抛出的各种网络/限流/参数异常，
    收敛成项目内部统一可识别的一种错误类型，方便 agent 层做降级处理。
    """

    def __init__(
        self,
        code: str,
        message: str,
        profile: str = "",
        provider: str = "",
        model: str = "",
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.profile = profile
        self.provider = provider
        self.model = model


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
    - 在 agent 层快速确认“当前这一跳到底走的是哪家模型”
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


def _resolve_embedding_cache_key(
    text: str,
    profile: str,
) -> tuple[str, str, str, str]:
    """生成 embedding 缓存 key，确保不同 provider/model/profile 不会串用。"""

    profile_config = _resolve_profile(profile, kind="embedding")
    provider_config = _resolve_provider(profile_config)
    model = _resolve_model(profile_config, provider_config)
    return (profile_config.name, provider_config.name, model, text)


def _get_cached_embedding(
    cache_key: tuple[str, str, str, str],
) -> list[float] | None:
    cache = _get_embedding_cache()
    embedding = cache.get(cache_key)
    stats = _get_embedding_cache_stats()
    if embedding is None:
        stats["misses"] += 1
        return None
    stats["hits"] += 1
    return list(embedding)


def _store_cached_embedding(
    cache_key: tuple[str, str, str, str],
    embedding: list[float],
) -> None:
    max_items = _get_embedding_cache_max_items()
    if max_items <= 0:
        return

    cache = _get_embedding_cache()
    if cache_key not in cache and len(cache) >= max_items:
        cache.clear()
    cache[cache_key] = list(embedding)
    _get_embedding_cache_stats()["stores"] += 1


def _get_client(provider: ProviderConfig) -> OpenAI:
    """获取（或创建并缓存）OpenAI 客户端。

    线程安全：使用 double-checked locking 模式
      1) 无锁快路径：命中即返回，避免热点争抢
      2) 加锁慢路径：再次检查，防止 TOCTOU 重复创建
    缓存 key 使用 API key 的 SHA256 前缀（不存明文），即使 cache key 被
    意外写入日志也不会泄漏凭证。
    """

    api_key_fingerprint = hashlib.sha256(
        (provider.api_key or "").encode("utf-8")
    ).hexdigest()[:16]
    cache_key = f"{provider.name}|{provider.base_url or ''}|{api_key_fingerprint}"

    # Fast path: 无锁读取，命中直接返回
    client = _CLIENT_CACHE.get(cache_key)
    if client is not None:
        return client

    # Slow path: 加锁后再次检查，避免并发重复创建
    with _CLIENT_CACHE_LOCK:
        client = _CLIENT_CACHE.get(cache_key)
        if client is not None:
            return client
        client = OpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            max_retries=0,
        )
        _CLIENT_CACHE[cache_key] = client
        return client


def _get_request_timeout(kind: str) -> float:
    if kind == "embedding":
        return _env_float(
            "EMBEDDING_TIMEOUT_SECONDS", DEFAULT_EMBEDDING_TIMEOUT_SECONDS
        )
    return _env_float("LLM_TIMEOUT_SECONDS", DEFAULT_CHAT_TIMEOUT_SECONDS)


def _get_max_retries(kind: str) -> int:
    if kind == "embedding":
        return max(0, _env_int("EMBEDDING_MAX_RETRIES", DEFAULT_LLM_MAX_RETRIES))
    return max(0, _env_int("LLM_MAX_RETRIES", DEFAULT_LLM_MAX_RETRIES))


def _get_retry_backoff_seconds(kind: str) -> float:
    if kind == "embedding":
        return max(
            0.0,
            _env_float(
                "EMBEDDING_RETRY_BACKOFF_SECONDS",
                DEFAULT_LLM_RETRY_BACKOFF_SECONDS,
            ),
        )
    return max(
        0.0,
        _env_float("LLM_RETRY_BACKOFF_SECONDS", DEFAULT_LLM_RETRY_BACKOFF_SECONDS),
    )


def _should_retry_exception(exc: Exception) -> bool:
    code = classify_exception(exc)
    return code not in NON_RETRYABLE_ERROR_CODES


def _raise_llm_call_error(
    exc: Exception,
    *,
    profile: str,
    provider: str,
    model: str,
) -> None:
    raise LLMCallError(
        code=classify_exception(exc),
        message=format_exception_message(exc),
        profile=profile,
        provider=provider,
        model=model,
    ) from exc


def _call_with_retry(
    operation,
    *,
    kind: str,
    profile: str,
    provider: str,
    model: str,
):
    """执行一次模型请求，带显式 timeout 和有限重试。

    SDK 自身重试关闭后，所有 chat/embedding 调用都走这里，避免网络抖动
    或限流时把整个 graph 永久挂住。
    """

    max_retries = _get_max_retries(kind)
    backoff_seconds = _get_retry_backoff_seconds(kind)
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _should_retry_exception(exc):
                break
            time.sleep(backoff_seconds * (2**attempt))

    _raise_llm_call_error(
        last_exc or RuntimeError("unknown llm call failure"),
        profile=profile,
        provider=provider,
        model=model,
    )


def _create_chat_completion(
    profile: str = PROFILE_DEFAULT_CHAT,
    trace_stage: str = "",
    **kwargs,
):
    """统一执行 chat completion 请求，并把底层 SDK 异常转换成项目内标准错误。"""

    profile_config = _resolve_profile(profile, kind="chat")
    provider_config = _resolve_provider(profile_config)
    model = kwargs.pop("model", None) or _resolve_model(profile_config, provider_config)
    client = _get_client(provider_config)
    call_index = next_model_call_index()
    timeout = _get_request_timeout("chat")
    max_retries = _get_max_retries("chat")
    add_current_run_metadata(
        build_model_trace_metadata(
            kind="chat",
            profile=profile_config.name,
            provider=provider_config.name,
            model=model,
            stage=trace_stage or profile_config.name,
            call_index=call_index,
            streaming=bool(kwargs.get("stream", False)),
            input_count=len(kwargs.get("messages") or []),
            timeout_seconds=timeout,
            max_retries=max_retries,
        ),
        event_name="model_call",
    )

    return _call_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            timeout=timeout,
            **kwargs,
        ),
        kind="chat",
        profile=profile_config.name,
        provider=provider_config.name,
        model=model,
    )


def _create_embedding(
    input: str | list[str],
    profile: str = PROFILE_DEFAULT_EMBEDDING,
):
    """统一执行 embedding 请求，并复用 provider/profile 解析逻辑。"""

    profile_config = _resolve_profile(profile, kind="embedding")
    provider_config = _resolve_provider(profile_config)
    model = _resolve_model(profile_config, provider_config)
    client = _get_client(provider_config)
    input_count = len(input) if isinstance(input, list) else 1
    call_index = next_model_call_index()
    timeout = _get_request_timeout("embedding")
    max_retries = _get_max_retries("embedding")
    add_current_run_metadata(
        build_model_trace_metadata(
            kind="embedding",
            profile=profile_config.name,
            provider=provider_config.name,
            model=model,
            stage=profile_config.name,
            call_index=call_index,
            input_count=input_count,
            timeout_seconds=timeout,
            max_retries=max_retries,
        ),
        event_name="embedding_call",
    )

    return _call_with_retry(
        lambda: client.embeddings.create(
            model=model,
            input=input,
            timeout=timeout,
        ),
        kind="embedding",
        profile=profile_config.name,
        provider=provider_config.name,
        model=model,
    )


def embed_text(text: str, profile: str = PROFILE_DEFAULT_EMBEDDING) -> list[float]:
    """单文本 embedding 封装。

    单条查询和老代码最常见，所以保留这个轻量接口，便于：
    - 检索 query 向量化
    - memory content 入库
    - 兼容旧版 `get_embedding(text)` 调用方式
    """

    cache_key = _resolve_embedding_cache_key(text, profile)
    cached_embedding = _get_cached_embedding(cache_key)
    if cached_embedding is not None:
        return cached_embedding

    response = _create_embedding(input=text, profile=profile)
    embedding = response.data[0].embedding
    _store_cached_embedding(cache_key, embedding)
    return list(embedding)


def embed_texts(
    texts: list[str],
    profile: str = PROFILE_DEFAULT_EMBEDDING,
) -> list[list[float]]:
    """批量 embedding 封装。

    给文档建索引这类场景预留统一入口，后面如果要做 batching、重试、缓存，
    只需要在这里增强，不用改业务层。
    """

    if not texts:
        return []

    results: list[list[float] | None] = [None] * len(texts)
    missing_texts: list[str] = []
    missing_keys: list[tuple[str, str, str, str]] = []
    pending_indexes_by_key: dict[tuple[str, str, str, str], list[int]] = {}

    for index, text in enumerate(texts):
        cache_key = _resolve_embedding_cache_key(text, profile)
        cached_embedding = _get_cached_embedding(cache_key)
        if cached_embedding is not None:
            results[index] = cached_embedding
            continue
        if cache_key in pending_indexes_by_key:
            pending_indexes_by_key[cache_key].append(index)
            continue
        pending_indexes_by_key[cache_key] = [index]
        missing_texts.append(text)
        missing_keys.append(cache_key)

    if missing_texts:
        response = _create_embedding(input=missing_texts, profile=profile)
        for offset, item in enumerate(response.data):
            embedding = list(item.embedding)
            cache_key = missing_keys[offset]
            for original_index in pending_indexes_by_key[cache_key]:
                results[original_index] = list(embedding)
            _store_cached_embedding(cache_key, embedding)

    return [item or [] for item in results]


def chat(
    messages: list[dict],
    max_completion_tokens: int | None = None,
    on_delta: Callable[[str], None] | None = None,
    profile: str = PROFILE_DEFAULT_CHAT,
    trace_stage: str = "",
) -> str:
    """项目内最通用的纯文本对话接口。

    - 非流式：直接返回完整字符串
    - 流式：边消费 SDK stream，边把 delta 回调给上层
    - profile：决定本次调用默认使用哪个 provider / model

    无论哪种模式，只要底层 SDK 调用失败，都会统一抛出 `LLMCallError`，
    由 agent 层决定是否降级、fallback 或返回用户可读提示。
    """
    request_kwargs = {
        "messages": messages,
    }
    if max_completion_tokens is not None:
        request_kwargs["max_completion_tokens"] = max_completion_tokens

    if on_delta is None:
        res = _create_chat_completion(
            profile=profile,
            trace_stage=trace_stage,
            **request_kwargs,
        )
        return (res.choices[0].message.content or "").strip()

    stream = _create_chat_completion(
        profile=profile,
        trace_stage=trace_stage,
        **request_kwargs,
        stream=True,
    )
    chunks: list[str] = []

    for event in stream:
        if not event.choices:
            continue

        delta = event.choices[0].delta.content or ""
        if not delta:
            continue

        chunks.append(delta)
        on_delta(delta)

    return "".join(chunks).strip()


def chat_with_tools(
    messages: list[dict],
    tools: list[dict],
    tool_impls: dict[str, Callable],
    max_completion_tokens: int | None = None,
    finalize_with_llm: bool = True,
    on_delta: Callable[[str], None] | None = None,
    profile: str = PROFILE_DEFAULT_CHAT,
) -> dict:
    """带 function calling 的统一工具对话接口。"""
    request_kwargs = {
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }
    if max_completion_tokens is not None:
        request_kwargs["max_completion_tokens"] = max_completion_tokens

    first_response = _create_chat_completion(
        profile=profile,
        trace_stage="tool_select",
        **request_kwargs,
    )
    first_message = first_response.choices[0].message
    tool_calls = first_message.tool_calls or []

    if not tool_calls:
        return {
            "answer": (first_message.content or "").strip(),
            "tool_calls": [],
            "tool_results": [],
        }

    followup_messages = list(messages)
    followup_messages.append(first_message.model_dump())
    executed_tool_calls = []
    tool_results = []

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        raw_arguments = tool_call.function.arguments or "{}"

        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            log_warning(
                "chat_with_tools",
                "invalid tool arguments JSON; fallback to empty arguments",
                {
                    "tool_name": tool_name,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "arguments_preview": raw_arguments[:160],
                },
            )
            arguments = {}

        tool_fn = tool_impls.get(tool_name)
        if tool_fn is None:
            tool_output = f"工具 {tool_name} 不存在。"
        else:
            try:
                tool_output = tool_fn(**arguments)
            except TypeError:
                tool_output = f"工具 {tool_name} 参数错误：{arguments}"
            except Exception as exc:
                tool_output = f"工具 {tool_name} 执行失败：{str(exc)}"

        executed_tool_calls.append(
            {
                "id": tool_call.id,
                "name": tool_name,
                "arguments": arguments,
            }
        )
        tool_results.append(
            {
                "id": tool_call.id,
                "name": tool_name,
                "output": tool_output,
            }
        )

        followup_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_output,
            }
        )

    final_answer = ""
    if finalize_with_llm:
        final_answer = chat(
            followup_messages,
            max_completion_tokens=max_completion_tokens,
            on_delta=on_delta,
            profile=profile,
            trace_stage="tool_finalize",
        )

    return {
        "answer": final_answer,
        "tool_calls": executed_tool_calls,
        "tool_results": tool_results,
    }


def rewrite_query(messages: list[dict], profile: str = PROFILE_REWRITE) -> str:
    """把上下文相关追问改写成适合检索的完整问题。"""
    res = _create_chat_completion(
        profile=profile,
        trace_stage="rewrite_query",
        messages=[
            {
                "role": "system",
                "content": REWRITE_QUERY_SYSTEM_PROMPT,
            },
            *messages,
        ],
    )
    return res.choices[0].message.content.strip()


def summarize_messages(
    old_summary: str, messages: list[dict], profile: str = PROFILE_SUMMARY
) -> str:
    """把旧摘要和新增对话压缩成新的会话状态摘要。"""
    latest_user_message = ""
    for m in reversed(messages):
        if m["role"] == "user":
            latest_user_message = m["content"]
            break

    recent_dialogue = "\n".join([f'{m["role"]}: {m["content"]}' for m in messages])

    res = _create_chat_completion(
        profile=profile,
        trace_stage="summarize_messages",
        messages=[
            {
                "role": "system",
                "content": SUMMARIZE_MESSAGES_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_summarize_messages_user_prompt(
                    old_summary=old_summary,
                    latest_user_message=latest_user_message,
                    recent_dialogue=recent_dialogue,
                ),
            },
        ],
    )

    return res.choices[0].message.content.strip()


def plan_routes(message: str, profile: str = PROFILE_ROUTING) -> list[str]:
    """根据用户问题为 supervisor 提供 agent 路由建议。"""
    try:
        res = _create_chat_completion(
            profile=profile,
            trace_stage="plan_routes",
            messages=[
                {
                    "role": "system",
                    "content": PLAN_ROUTES_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_route_planning_user_prompt(message),
                },
            ],
        )
    except LLMCallError as exc:
        log_warning(
            "plan_routes",
            "route planning LLM failed; fallback to chat_agent",
            {
                "code": exc.code,
                "profile": exc.profile,
                "provider": exc.provider,
                "model": exc.model,
            },
        )
        return [ROUTE_CHAT_AGENT]

    text = res.choices[0].message.content.strip()

    try:
        routes = json.loads(text)
        allowed = {
            ROUTE_TOOL_AGENT,
            ROUTE_RAG_AGENT,
            ROUTE_CHAT_AGENT,
            ROUTE_NOVEL_SCRIPT_AGENT,
        }
        routes = [r for r in routes if r in allowed]
        return routes or [ROUTE_CHAT_AGENT]
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        log_warning(
            "plan_routes",
            "invalid route planning response; fallback to chat_agent",
            {
                "error": f"{exc.__class__.__name__}: {exc}",
                "response_preview": text[:160],
            },
        )
        return [ROUTE_CHAT_AGENT]

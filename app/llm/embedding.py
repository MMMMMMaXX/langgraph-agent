"""Embedding 相关接口。

- `_create_embedding`：底层 embedding 请求封装
- `embed_text` / `embed_texts`：带请求级缓存的单条 / 批量入口
"""

from app.constants.model_profiles import PROFILE_DEFAULT_EMBEDDING
from app.llm.caching import (
    _get_cached_embedding,
    _get_client,
    _resolve_embedding_cache_key,
    _store_cached_embedding,
)
from app.llm.providers import (
    _resolve_model,
    _resolve_profile,
    _resolve_provider,
)
from app.llm.retry import (
    _call_with_retry,
    _get_max_retries,
    _get_request_timeout,
)
from app.tracing import (
    add_current_run_metadata,
    build_model_trace_metadata,
    next_model_call_index,
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

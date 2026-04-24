"""两类独立缓存：
1. Embedding 请求级缓存（ContextVar 驱动，每个请求隔离，用于去重同轮内的 embedding）
2. OpenAI 客户端进程级缓存（threading.Lock 保护，避免高并发重复建连）

两类缓存互不耦合，但放在同一模块里方便统一 reset/debug。
"""

import hashlib
import threading
from contextvars import ContextVar

from openai import OpenAI

from app.constants.model_profiles import ENV_EMBEDDING_CACHE_MAX_ITEMS
from app.llm._helpers import _env_int
from app.llm.providers import (
    ProviderConfig,
    _resolve_model,
    _resolve_profile,
    _resolve_provider,
)

# --- Embedding 缓存（ContextVar：按请求/协程隔离，天生无竞态） ----------------

# 单次请求内最多缓存多少条 embedding，避免异常长请求导致内存无限增长。
DEFAULT_EMBEDDING_CACHE_MAX_ITEMS = 128

_EMBEDDING_CACHE: ContextVar[dict[tuple[str, str, str, str], list[float]] | None] = (
    ContextVar("embedding_cache", default=None)
)
_EMBEDDING_CACHE_STATS: ContextVar[dict[str, int] | None] = ContextVar(
    "embedding_cache_stats",
    default=None,
)


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


# --- OpenAI 客户端缓存（threading.Lock：进程级共享，有竞态需要保护） ---------

_CLIENT_CACHE: dict[str, OpenAI] = {}
# 客户端缓存锁：防止高并发下多个线程同时 miss 后重复创建 OpenAI client，
# 造成连接池浪费。锁只保护字典读写本身，不覆盖任何业务调用。
_CLIENT_CACHE_LOCK = threading.Lock()


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

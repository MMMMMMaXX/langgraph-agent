import os
from dataclasses import dataclass

from app.env import load_project_env

load_project_env()


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def get_env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


@dataclass(frozen=True)
class RagConfig:
    doc_top_k: int = 5
    doc_score_threshold: float = 0.5
    doc_soft_match_threshold: float = 0.35
    doc_rerank_top_k: int = 2
    memory_top_k: int = 5
    memory_rerank_top_k: int = 5
    max_doc_context_chars: int = 360
    max_doc_context_blocks: int = 2
    max_doc_answer_tokens: int = 180
    max_memory_answer_tokens: int = 140


@dataclass(frozen=True)
class MemoryConfig:
    max_recent_messages: int = 4
    summary_trigger: int = 6


@dataclass(frozen=True)
class ConversationHistoryConfig:
    """非向量化会话流水配置。

    这份历史只用于“刚才问了什么 / 总结所有问题”这类顺序型回放，
    不参与语义检索，也不替代 Chroma memory。
    """

    backend: str = "sqlite"
    path: str = "data/conversation_history.sqlite3"
    sqlite_path: str = "data/conversation_history.sqlite3"
    jsonl_path: str = "data/conversation_history.jsonl"
    recent_limit: int = 5
    all_limit: int = 20
    dedupe_window_seconds: int = 600


@dataclass(frozen=True)
class VectorStoreConfig:
    """向量存储配置。

    当前阶段采用“项目外部算 embedding，Chroma 只负责存储和检索”的方案，
    因此这里主要管理：
    - Chroma 本地持久化目录
    - docs / memory 的 collection 名称
    - 迁移开关
    """

    provider: str = "json"
    chroma_persist_dir: str = "data/chroma"
    doc_collection_name: str = "docs"
    memory_collection_name: str = "memory"


@dataclass(frozen=True)
class ChunkingConfig:
    """文档切块配置。

    当前只对 doc 做 chunk；memory 暂时保持整条存储，不做拆分。
    这里先采用偏保守的字符数切块策略，优先保证：
    - 实现简单
    - 调试成本低
    - 后续容易替换成更复杂的语义切块
    """

    enabled: bool = True
    chunk_size_chars: int = 280
    chunk_overlap_chars: int = 60
    min_chunk_chars: int = 40


@dataclass(frozen=True)
class CheckpointConfig:
    """LangGraph checkpoint 配置。

    checkpoint 保存 graph state，用于 session 恢复；
    conversation_history 保存业务流水，两者不要混用。
    """

    enabled: bool = True
    path: str = "data/langgraph_checkpoints.sqlite3"


def load_rag_config() -> RagConfig:
    # 配置优先从环境变量读取，方便做对照实验时不改代码。
    return RagConfig(
        doc_top_k=get_env_int("DOC_TOP_K", 5),
        doc_score_threshold=get_env_float("DOC_SCORE_THRESHOLD", 0.5),
        doc_soft_match_threshold=get_env_float("DOC_SOFT_MATCH_THRESHOLD", 0.35),
        doc_rerank_top_k=get_env_int("DOC_RERANK_TOP_K", 2),
        memory_top_k=get_env_int("RAG_MEMORY_TOP_K", 5),
        memory_rerank_top_k=get_env_int("RAG_MEMORY_RERANK_TOP_K", 5),
        max_doc_context_chars=get_env_int("MAX_DOC_CONTEXT_CHARS", 360),
        max_doc_context_blocks=get_env_int("MAX_DOC_CONTEXT_BLOCKS", 2),
        max_doc_answer_tokens=get_env_int("MAX_DOC_ANSWER_TOKENS", 180),
        max_memory_answer_tokens=get_env_int("MAX_MEMORY_ANSWER_TOKENS", 140),
    )


def load_memory_config() -> MemoryConfig:
    return MemoryConfig(
        max_recent_messages=get_env_int("MAX_RECENT_MESSAGES", 4),
        summary_trigger=get_env_int("SUMMARY_TRIGGER", 6),
    )


def load_conversation_history_config() -> ConversationHistoryConfig:
    legacy_path = get_env_str("CONVERSATION_HISTORY_PATH", "")
    sqlite_path = get_env_str(
        "CONVERSATION_HISTORY_SQLITE_PATH",
        (
            legacy_path
            if legacy_path.endswith((".sqlite", ".sqlite3", ".db"))
            else "data/conversation_history.sqlite3"
        ),
    )
    jsonl_path = get_env_str(
        "CONVERSATION_HISTORY_JSONL_PATH",
        (
            legacy_path
            if legacy_path.endswith(".jsonl")
            else "data/conversation_history.jsonl"
        ),
    )
    backend = get_env_str("CONVERSATION_HISTORY_BACKEND", "sqlite").lower()
    default_path = sqlite_path if backend == "sqlite" else jsonl_path
    return ConversationHistoryConfig(
        backend=backend,
        path=get_env_str("CONVERSATION_HISTORY_PATH", default_path),
        sqlite_path=sqlite_path,
        jsonl_path=jsonl_path,
        recent_limit=get_env_int("CONVERSATION_HISTORY_RECENT_LIMIT", 5),
        all_limit=get_env_int("CONVERSATION_HISTORY_ALL_LIMIT", 20),
        dedupe_window_seconds=get_env_int("CONVERSATION_HISTORY_DEDUPE_SECONDS", 600),
    )


def load_vector_store_config() -> VectorStoreConfig:
    return VectorStoreConfig(
        provider=get_env_str("VECTOR_STORE_PROVIDER", "json"),
        chroma_persist_dir=get_env_str("CHROMA_PERSIST_DIR", "data/chroma"),
        doc_collection_name=get_env_str("CHROMA_DOC_COLLECTION", "docs"),
        memory_collection_name=get_env_str("CHROMA_MEMORY_COLLECTION", "memory"),
    )


def load_chunking_config() -> ChunkingConfig:
    return ChunkingConfig(
        enabled=get_env_bool("DOC_CHUNKING_ENABLED", True),
        chunk_size_chars=get_env_int("DOC_CHUNK_SIZE_CHARS", 280),
        chunk_overlap_chars=get_env_int("DOC_CHUNK_OVERLAP_CHARS", 60),
        min_chunk_chars=get_env_int("DOC_MIN_CHUNK_CHARS", 40),
    )


def load_checkpoint_config() -> CheckpointConfig:
    return CheckpointConfig(
        enabled=get_env_bool("LANGGRAPH_CHECKPOINT_ENABLED", True),
        path=get_env_str(
            "LANGGRAPH_CHECKPOINT_SQLITE_PATH",
            "data/langgraph_checkpoints.sqlite3",
        ),
    )


# 先收敛最常调的实验参数，后续再逐步扩大配置覆盖面。
RAG_CONFIG = load_rag_config()
MEMORY_CONFIG = load_memory_config()
CONVERSATION_HISTORY_CONFIG = load_conversation_history_config()
VECTOR_STORE_CONFIG = load_vector_store_config()
CHUNKING_CONFIG = load_chunking_config()
CHECKPOINT_CONFIG = load_checkpoint_config()

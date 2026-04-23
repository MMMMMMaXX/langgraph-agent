"""RAG agent 内部数据结构。"""

from dataclasses import dataclass


@dataclass
class DocRetrievalResult:
    """文档检索阶段的完整结果。"""

    docs: list[dict]
    filtered_docs: list[dict]
    doc_hits: list[dict]
    merged_doc_hits: list[dict]
    retrieval_debug: dict
    errors: list[str]
    timings_ms: dict[str, float]


@dataclass
class MemoryRetrievalResult:
    """RAG memory 兜底检索阶段的完整结果。"""

    memory_hits: list[dict]
    memory_before_rerank: list[dict]
    retrieval_debug: dict
    errors: list[str]
    timings_ms: dict[str, float]


@dataclass
class RagContext:
    """送入回答生成模型的上下文集合。"""

    context: str
    doc_context: str
    memory_context: str


@dataclass
class RewriteResult:
    """RAG 查询改写结果。"""

    query: str
    errors: list[str]
    timing_ms: float


@dataclass
class RagAnswerResult:
    """RAG 回答生成结果。"""

    answer: str
    errors: list[str]
    timing_ms: float

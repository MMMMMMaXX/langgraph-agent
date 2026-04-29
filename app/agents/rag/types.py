"""RAG agent 内部数据结构。"""

from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class DocRetrievalPipelineConfig:
    """文档检索 pipeline 配置。"""

    query_type: str
    doc_top_k: int
    doc_rerank_top_k: int
    candidate_top_k: int
    score_threshold: float
    soft_match_threshold: float
    hybrid_alpha: float
    hybrid_beta: float
    dense_enabled: bool = True
    lexical_enabled: bool = True
    rerank_enabled: bool = True
    chunk_merge_enabled: bool = True
    source_diversity_enabled: bool = False


@dataclass
class DocRetrievalPipelineState:
    """文档检索 pipeline 中间态。"""

    query: str
    config: DocRetrievalPipelineConfig
    dense_hits: list[dict]
    lexical_hits: list[dict]
    hybrid_hits: list[dict]
    docs: list[dict]
    filtered_docs: list[dict]
    doc_hits: list[dict]
    merged_doc_hits: list[dict]
    diversified_doc_hits: list[dict]
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
    citations: list[dict]
    context_compression: dict = field(default_factory=dict)
    memory_compression: dict = field(default_factory=dict)


@dataclass
class RewriteResult:
    """RAG 查询改写结果。"""

    query: str
    errors: list[str]
    timing_ms: float
    mode: str = "default"
    trigger: str = ""
    skipped_reason: str = ""


@dataclass
class QueryClassification:
    """RAG query 分类结果。"""

    query_type: str
    confidence: float
    reason: str


@dataclass
class RagAnswerResult:
    """RAG 回答生成结果。"""

    answer: str
    errors: list[str]
    timing_ms: float

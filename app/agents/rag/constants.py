"""rag_agent 专属常量。

这些常量只服务于 RAG agent 的检索、调试和回答策略，先集中放在这里，
避免散落在 node 主流程里。后续如果多个 agent 复用，再上移到 app/constants。
"""

# 定义类问题送入模型的最大上下文字符数，减少短问答的 token 和延迟。
DEFINITION_CONTEXT_CHARS = 280

# 定义类问题的最大回答 token，约束输出为短定义。
DEFINITION_MAX_ANSWER_TOKENS = 100

# 少量候选时才允许跳过 LLM rerank，避免误跳过复杂候选集。
DOC_RERANK_SKIP_MAX_CANDIDATES = 2

# 候选分差小于该值时，认为相邻 chunk 的排序不需要再交给 LLM 判断。
DOC_RERANK_SKIP_SCORE_DELTA = 0.05

# 同时命中 dense 和 keyword 时，认为召回来源更可靠。
HIGH_CONFIDENCE_RETRIEVAL_SOURCES = {"dense", "keyword"}

# 合并相邻 chunk 时使用的段落分隔符。
MERGED_CHUNK_SEPARATOR = "\n"

# debug 中单条文档命中的预览字符数。
DOC_HIT_DEBUG_TEXT_PREVIEW_CHARS = 60

# hybrid 打分 debug 中的内容预览字符数。
HYBRID_DOC_TEXT_PREVIEW_CHARS = 40

# 文档命中 debug 允许透出的稳定字段。
DOC_HIT_DEBUG_FIELDS = (
    "id",
    "doc_id",
    "doc_title",
    "source",
    "chunk_index",
    "start_char",
    "end_char",
    "chunk_char_len",
    "retrieval_source",
    "retrieval_sources",
    "merged_chunk_ids",
    "merged_chunk_indexes",
    "score",
    "semantic_score",
    "keyword_score_norm",
)

# 文档命中 debug 中需要统一保留 4 位小数的分数字段。
DOC_HIT_SCORE_FIELDS = {
    "score",
    "semantic_score",
    "keyword_score_norm",
}


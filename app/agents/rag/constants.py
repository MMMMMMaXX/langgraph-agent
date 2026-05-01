"""rag_agent 专属常量。

这些常量只服务于 RAG agent 的检索、调试和回答策略，先集中放在这里，
避免散落在 node 主流程里。后续如果多个 agent 复用，再上移到 app/constants。
"""

from app.constants.retrieval import HIGH_CONFIDENCE_RETRIEVAL_SOURCES

# 定义类问题送入模型的最大上下文字符数，减少短问答的 token 和延迟。
DEFINITION_CONTEXT_CHARS = 280

# 定义类问题的最大回答 token，约束输出为短定义。
DEFINITION_MAX_ANSWER_TOKENS = 100

# Query 类型：定义类问题，如“WAI-ARIA 是什么”。
QUERY_TYPE_DEFINITION = "definition"

# Query 类型：对比类问题，如“A 和 B 有什么区别”。
QUERY_TYPE_COMPARISON = "comparison"

# Query 类型：追问类问题，如“那上海呢 / 它和前者相比呢”。
QUERY_TYPE_FOLLOWUP = "followup"

# Query 类型：普通事实/知识问答。
QUERY_TYPE_FACTUAL = "factual"

# Query 类型：信息过少或指代不明，适合低置信兜底。
QUERY_TYPE_FALLBACK = "fallback"

# 对比类回答策略：保留更多上下文，要求输出差异点。
ANSWER_STRATEGY_COMPARISON = "comparison"

# 追问类回答策略：更强调结合改写后的完整问题回答。
ANSWER_STRATEGY_FOLLOWUP = "followup"

# 低置信兜底策略：明确资料不足时不要硬答。
ANSWER_STRATEGY_FALLBACK = "fallback"

# 少量候选时才允许跳过 LLM rerank，避免误跳过复杂候选集。
DOC_RERANK_SKIP_MAX_CANDIDATES = 2

# 候选分差小于该值时，认为相邻 chunk 的排序不需要再交给 LLM 判断。
DOC_RERANK_SKIP_SCORE_DELTA = 0.05

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

# Context compression：句子切分后最多保留的候选句数量，控制规则压缩复杂度。
CONTEXT_COMPRESSION_MAX_SENTENCES_PER_BLOCK = 4

# Context compression：每个引用块至少保留的正文字符数，避免压缩后证据过短。
CONTEXT_COMPRESSION_MIN_BLOCK_CHARS = 80

# Context compression：定义类问题中可加权的定义/作用信号词。
CONTEXT_COMPRESSION_DEFINITION_SIGNALS = (
    "是",
    "指",
    "称为",
    "定义",
    "用于",
    "作用",
    "提供",
)

# Memory context compression：每条记忆命中最多保留的字符数（约 2-3 句），
# 控制单条 memory 过长时对 context 的冲击。
MEMORY_COMPRESSION_MAX_BLOCK_CHARS = 120

# Memory context compression：memory 上下文总字符上限，
# 防止多轮对话积累的记忆内容撑满 token 预算。
MEMORY_COMPRESSION_MAX_TOTAL_CHARS = 360

# Memory context compression：最多纳入上下文的记忆命中条数，
# 避免低质尾部条目引入噪声。
MEMORY_COMPRESSION_MAX_HITS = 3

# 动态 max_tokens：单次答案允许的最小 token 下限，
# 避免 context 极短时答案被压缩到无法成句。
ANSWER_TOKENS_MIN = 40

# 动态 max_tokens：context 字符数到答案 token 上限的换算比例。
# 中文约 1.5~2 chars/token，取 2 作保守值：360 chars → 180 tokens。
# 实际上限 = max(ANSWER_TOKENS_MIN, min(策略上限, context_chars // RATIO))
ANSWER_TOKENS_CONTEXT_RATIO = 2

# LLM 分类器：规则置信度低于此值时触发 LLM 二裁。
# FACTUAL 默认 0.6、FALLBACK 默认 0.75，均低于阈值；高置信 DEFINITION/FOLLOWUP 不受影响。
CLASSIFIER_LLM_CONFIDENCE_THRESHOLD = 0.8

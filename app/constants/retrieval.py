"""检索链路共享常量。"""

# Dense / lexical 召回来源标识：会写入 retrieval_sources、debug 和 rerank prompt。
RETRIEVAL_SOURCE_DENSE = "dense"
RETRIEVAL_SOURCE_KEYWORD = "keyword"
HIGH_CONFIDENCE_RETRIEVAL_SOURCES = {
    RETRIEVAL_SOURCE_DENSE,
    RETRIEVAL_SOURCE_KEYWORD,
}

# Hybrid 融合默认权重：alpha=dense semantic，beta=lexical keyword。
DEFAULT_HYBRID_ALPHA = 0.65
DEFAULT_HYBRID_BETA = 0.35

# Dense / lexical 初召回候选放大倍数，给后续 hybrid/rerank 留排序空间。
DOC_CANDIDATE_MULTIPLIER = 4

# Lexical query 停用词：这些词常表达提问方式而非知识实体，参与 FTS OR 查询时
# 容易把“什么时候/应该/使用”这类泛词段落排到精确实体（如“脚本”）前面。
LEXICAL_QUERY_STOPWORDS = {
    "是什么",
    "什么是",
    "为什么",
    "什么时候",
    "什么",
    "时候",
    "怎么",
    "如何",
    "应该",
    "使用",
    "一下",
    "这个",
    "那个",
}

# Lexical rescue 最低关键词归一分：候选整体 hybrid 分不足 0.5，但命中查询中的
# 关键中文词时，允许进入 rerank，避免“脚本”这类精确词 chunk 在 threshold 阶段被丢掉。
LEXICAL_RESCUE_MIN_KEYWORD_SCORE = 0.3

# Lexical rescue 最大补充数量：只救少量精确词候选，防止低分长尾撑爆 rerank/context。
LEXICAL_RESCUE_MAX_DOCS = 2

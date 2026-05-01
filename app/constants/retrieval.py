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

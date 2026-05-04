"""知识库管理相关常量。

这里集中放知识库 API / 管理工具共用的非模型业务常量，避免路由层、service
和测试里散落 magic number。真正可通过环境变量调节的运行时参数仍放
app.config。
"""

# Rechunk preview：默认展示的样例 chunk 数。
RECHUNK_PREVIEW_DEFAULT_SAMPLE_LIMIT = 5

# Rechunk preview：最多展示的样例 chunk 数，防止 API 响应过大。
RECHUNK_PREVIEW_MAX_SAMPLE_LIMIT = 50

# Rechunk preview：允许预览的最小 chunk size。
RECHUNK_PREVIEW_MIN_CHUNK_SIZE_CHARS = 40

# Rechunk preview：允许预览的最大 chunk size，过大会让 dry-run 失去诊断意义。
RECHUNK_PREVIEW_MAX_CHUNK_SIZE_CHARS = 5000

# Rechunk preview：允许预览的最小 overlap。
RECHUNK_PREVIEW_MIN_OVERLAP_CHARS = 0

# Rechunk preview：允许预览的最大 overlap。
RECHUNK_PREVIEW_MAX_OVERLAP_CHARS = 1000

# Rechunk preview：允许预览的最小 chunk 保留长度。
RECHUNK_PREVIEW_MIN_MIN_CHUNK_CHARS = 1

# Rechunk preview：允许预览的最大 chunk 保留长度。
RECHUNK_PREVIEW_MAX_MIN_CHUNK_CHARS = 2000

# Rechunk preview source mode：当前 catalog 尚未存完整原文，先从已有 chunks 近似重建。
RECHUNK_SOURCE_MODE_RECONSTRUCTED_FROM_CHUNKS = "reconstructed_from_chunks"

# Rechunk preview source mode：未来 catalog 存完整原文后使用该模式。
RECHUNK_SOURCE_MODE_DOCUMENT_CONTENT = "document_content"

# Rechunk preview warning：使用 chunk 拼接文本，不是严格原文。
RECHUNK_WARNING_SOURCE_RECONSTRUCTED = "source_reconstructed_from_chunks"

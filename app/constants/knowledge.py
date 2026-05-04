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

# Rechunk preview warning：候选参数没有切出任何 chunk，通常说明 min_chunk/chunk_size 不合理。
RECHUNK_WARNING_PREVIEW_GENERATED_NO_CHUNKS = "preview_generated_no_chunks"

# Knowledge catalog documents 表：保存导入后规范化原文的列名。
DOCUMENT_CONTENT_TEXT_COLUMN = "content_text"

# Knowledge catalog documents 表：保存规范化原文字符数，列表/详情默认只暴露这个轻量指标。
DOCUMENT_CONTENT_CHAR_LEN_COLUMN = "content_char_len"

# Knowledge catalog documents 表：记录解析器名称，为后续 PDF/DOCX/HTML parser 版本化预留。
DOCUMENT_PARSER_NAME_COLUMN = "parser_name"

# Knowledge catalog documents 表：记录解析器版本，方便未来排查“同一文档不同解析结果”。
DOCUMENT_PARSER_VERSION_COLUMN = "parser_version"

# 默认 parser：当前导入链路接收的已经是纯文本/Markdown/JSON 抽取后的正文。
DEFAULT_DOCUMENT_PARSER_NAME = "raw_text"

# 默认 parser 版本：后续解析规则变化时递增，便于判断是否需要重新导入或重建索引。
DEFAULT_DOCUMENT_PARSER_VERSION = "v1"

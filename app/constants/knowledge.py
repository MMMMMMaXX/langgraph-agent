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

# Rechunk 错误：指定文档不存在，路由层会映射为 404。
RECHUNK_ERROR_DOCUMENT_NOT_FOUND = "document not found"

# Rechunk apply 错误：旧数据没有完整原文，不能安全地执行真实重切片。
RECHUNK_ERROR_DOCUMENT_CONTENT_MISSING = "document content not available; reimport document before apply"

# Rechunk apply 日志 stage：新 Chroma 重建失败后，已经尝试恢复旧 SQLite/FTS5。
RECHUNK_APPLY_ROLLBACK_STAGE = "knowledge.rechunk.apply.rollback"

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

# Knowledge API：文档列表默认返回数量。
KNOWLEDGE_DOC_LIST_DEFAULT_LIMIT = 50

# Knowledge API：文档列表最大返回数量，避免管理页面误拉过多数据。
KNOWLEDGE_DOC_LIST_MAX_LIMIT = 200

# Knowledge API：chunk 列表默认返回数量，适合普通文档的首屏调试。
KNOWLEDGE_CHUNK_LIST_DEFAULT_LIMIT = 50

# Knowledge API：chunk 列表最大返回数量，避免长文档一次性撑爆响应。
KNOWLEDGE_CHUNK_LIST_MAX_LIMIT = 500

# Knowledge API：chunk 内容预览默认字符数，只展示片段，不返回完整大文本。
KNOWLEDGE_CHUNK_PREVIEW_DEFAULT_CHARS = 220

# Knowledge API：chunk 内容预览最大字符数，上限用于保护接口响应体积。
KNOWLEDGE_CHUNK_PREVIEW_MAX_CHARS = 2000

# Knowledge health：最多精确对比多少条 Chroma chunk id；超过后只做数量级检查。
KNOWLEDGE_HEALTH_MAX_EXACT_CHROMA_CHECK = 5000

# Knowledge health 状态：SQLite/FTS/Chroma 都可用且精确一致。
KNOWLEDGE_HEALTH_STATUS_OK = "ok"

# Knowledge health 状态：核心可用，但存在数量不一致或检查降级。
KNOWLEDGE_HEALTH_STATUS_WARN = "warn"

# Knowledge health 状态：核心检查异常，知识库需要人工处理。
KNOWLEDGE_HEALTH_STATUS_ERROR = "error"

# Knowledge health warning：Chroma 与 SQLite chunk 数量不一致。
KNOWLEDGE_HEALTH_WARNING_CHROMA_COUNT_MISMATCH = "chroma_count_mismatch"

# Knowledge health warning：FTS5 与 SQLite chunk 数量不一致。
KNOWLEDGE_HEALTH_WARNING_FTS_COUNT_MISMATCH = "fts_count_mismatch"

# Knowledge health warning：Chroma 数量超过精确 id 对比上限。
KNOWLEDGE_HEALTH_WARNING_EXACT_CHECK_SKIPPED = "exact_chroma_id_check_skipped"

# Knowledge health warning：Chroma 中存在 SQLite catalog 没有的 chunk id。
KNOWLEDGE_HEALTH_WARNING_ORPHAN_CHROMA_CHUNKS = "orphan_chroma_chunks"

# Knowledge health warning：SQLite catalog 中存在没有写入 Chroma 的 chunk id。
KNOWLEDGE_HEALTH_WARNING_MISSING_CHROMA_CHUNKS = "missing_chroma_chunks"

# Knowledge API 错误：指定文档不存在，路由层统一复用。
KNOWLEDGE_ERROR_DOCUMENT_NOT_FOUND = "document not found"

"""Eval runner 相关常量。

这些常量只服务本地评测脚本和测试，集中在这里是为了避免 eval case key、
输出占位符和布尔字符串散在脚本各处。
"""

# Eval 输出占位：表示该指标不适用于当前 case。
EVAL_FIELD_NOT_APPLICABLE = "-"

# Eval 布尔文本：CSV/表格中统一使用小写字符串，便于后续脚本聚合。
EVAL_BOOL_TRUE = "true"

# Eval 布尔文本：CSV/表格中统一使用小写字符串，便于后续脚本聚合。
EVAL_BOOL_FALSE = "false"

# Eval case key：声明 case 运行前需要导入的临时知识库文档列表。
EVAL_CASE_SETUP_IMPORTS_KEY = "setup_knowledge_imports"

# Eval import key：导入文档别名，用于 expected_import_aliases 解析动态 doc_id。
EVAL_IMPORT_ALIAS_KEY = "alias"

# Eval import key：直接内联正文，适合极小测试文档。
EVAL_IMPORT_CONTENT_KEY = "content"

# Eval import key：从仓库内文件读取正文，适合真实/较长文档 fixture。
EVAL_IMPORT_CONTENT_PATH_KEY = "content_path"

# Eval case key：声明期望命中的“导入文档 alias + chunk_index”组合。
EVAL_EXPECTED_IMPORT_CHUNKS_KEY = "expected_import_chunks"

# Eval expected_import_chunks key：引用 setup_knowledge_imports 中的导入别名。
EVAL_EXPECTED_IMPORT_CHUNK_ALIAS_KEY = "alias"

# Eval expected_import_chunks key：导入文档内期望命中的 chunk 序号。
EVAL_EXPECTED_IMPORT_CHUNK_INDEX_KEY = "chunk_index"

# Eval case key：声明该 case 期望走知识不足 fallback。
EVAL_EXPECTED_FALLBACK_KEY = "expected_fallback"

# Eval category：知识不足 fallback 用例分类。
EVAL_CATEGORY_FALLBACK = "fallback"

# Eval 输出环境变量：指定 JSON 结果文件路径。
EVAL_OUTPUT_JSON_ENV = "EVAL_OUTPUT_JSON"

# Eval 输出环境变量：指定 CSV 结果文件路径。
EVAL_OUTPUT_CSV_ENV = "EVAL_OUTPUT_CSV"

# Eval 输入环境变量：指定要连接的已启动 API 地址；为空时使用进程内 TestClient。
EVAL_BASE_URL_ENV = "EVAL_BASE_URL"

# Eval 输入环境变量：用逗号分隔要运行的 case id。
EVAL_CASE_IDS_ENV = "EVAL_CASE_IDS"

# Eval 输入环境变量：HTTP 模式请求已启动 API 的超时时间，单位秒。
EVAL_HTTP_TIMEOUT_ENV = "EVAL_HTTP_TIMEOUT"

# Eval history 环境变量：eval_chat 读取该路径来清理/隔离评测会话历史。
EVAL_CONVERSATION_HISTORY_PATH_ENV = "EVAL_CONVERSATION_HISTORY_PATH"

# Eval HTTP 默认超时秒数：长文本/创作类 case 可能比普通 QA 慢很多。
EVAL_HTTP_TIMEOUT_SECONDS = 120.0

# Runtime history 环境变量：应用层通用历史路径配置。
CONVERSATION_HISTORY_PATH_ENV = "CONVERSATION_HISTORY_PATH"

# Runtime history 环境变量：应用层历史后端类型配置。
CONVERSATION_HISTORY_BACKEND_ENV = "CONVERSATION_HISTORY_BACKEND"

# Runtime history 环境变量值：SQLite 历史后端。
CONVERSATION_HISTORY_BACKEND_SQLITE = "sqlite"

# Runtime history 环境变量：SQLite 历史库路径配置。
CONVERSATION_HISTORY_SQLITE_PATH_ENV = "CONVERSATION_HISTORY_SQLITE_PATH"

# Runtime knowledge 环境变量：知识库 catalog SQLite 路径配置。
KNOWLEDGE_BASE_SQLITE_PATH_ENV = "KNOWLEDGE_BASE_SQLITE_PATH"

# Runtime vector 环境变量：Chroma 持久化目录配置。
CHROMA_PERSIST_DIR_ENV = "CHROMA_PERSIST_DIR"

# Eval 输出后缀：每次评测独立的对话历史 SQLite。
EVAL_CONVERSATION_HISTORY_SUFFIX = ".conversation_history.sqlite3"

# Eval 输出后缀：每次评测独立的知识库 catalog SQLite。
EVAL_KNOWLEDGE_SQLITE_SUFFIX = ".knowledge.sqlite3"

# Eval 输出后缀：每次评测独立的 Chroma 持久化目录。
EVAL_CHROMA_DIR_SUFFIX = ".chroma"

# Eval 输出后缀：每次评测的 manifest（运行元信息 + chroma 生命周期记录）。
EVAL_MANIFEST_SUFFIX = ".manifest.json"

# Eval 环境变量：保留 eval 自动创建的 Chroma 目录，便于失败排查。
EVAL_KEEP_CHROMA_ENV = "EVAL_KEEP_CHROMA"

# Eval manifest 字段：Chroma 目录路径与生命周期。
EVAL_MANIFEST_CHROMA_PERSIST_DIR_KEY = "chroma_persist_dir"
EVAL_MANIFEST_CHROMA_AUTO_CREATED_KEY = "chroma_auto_created"
EVAL_MANIFEST_CHROMA_CLEANED_KEY = "chroma_cleaned"
EVAL_MANIFEST_CHROMA_KEEP_REASON_KEY = "chroma_keep_reason"
EVAL_MANIFEST_CHROMA_SIZE_BYTES_KEY = "chroma_size_bytes"
# wrapper 在清理前/后分别记录目录大小：清理后会被覆写为 0，
# 但仍想知道清理前 Chroma 真实占用，便于排查（chunks 与磁盘占用对比）。
EVAL_MANIFEST_CHROMA_SIZE_BEFORE_KEY = "chroma_size_bytes_before_cleanup"
EVAL_MANIFEST_CHROMA_SIZE_AFTER_KEY = "chroma_size_bytes_after_cleanup"
EVAL_MANIFEST_RUN_STATUS_KEY = "run_status"
EVAL_MANIFEST_RUN_STATUS_SUCCESS = "success"
EVAL_MANIFEST_RUN_STATUS_FAILURE = "failure"
EVAL_MANIFEST_RUN_STATUS_ERROR = "error"
EVAL_MANIFEST_PASS_RATE_KEY = "pass_rate"
EVAL_MANIFEST_TOTAL_KEY = "total_cases"
EVAL_MANIFEST_FAILED_KEY = "failed_cases"

# Chroma 保留原因常量。
EVAL_CHROMA_KEEP_REASON_FAILURE = "failure"
EVAL_CHROMA_KEEP_REASON_FLAG = "keep_flag"
EVAL_CHROMA_KEEP_REASON_EXTERNAL = "external_persist_dir"

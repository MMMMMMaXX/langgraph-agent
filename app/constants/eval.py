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

# Eval case key：声明该 case 期望走知识不足 fallback。
EVAL_EXPECTED_FALLBACK_KEY = "expected_fallback"

# Eval category：知识不足 fallback 用例分类。
EVAL_CATEGORY_FALLBACK = "fallback"

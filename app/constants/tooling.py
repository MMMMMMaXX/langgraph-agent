"""工具调用相关常量。"""

# 天气工具名称，必须和 function calling schema 中的 name 一致。
TOOL_NAME_GET_WEATHER = "get_weather"

# 计算工具名称，必须和 function calling schema 中的 name 一致。
TOOL_NAME_CALCULATE = "calculate"

# 创建工单工具名称（side_effect 类）；下游 mock 表 `mock_tickets`。
# 用下划线而非点号：OpenAI function-calling 要求 name 匹配
# `^[a-zA-Z0-9_-]+$`，有点号会被 API 直接拒绝。
TOOL_NAME_TICKET_CREATE = "ticket_create"

# 查询 service 监控错误日志（read_only mock 工具，Phase 2 workflow demo 用）。
# 业务名 `monitor.query_errors`，function name `monitor_query_errors`；
# ToolRegistry 会把业务名映射到 function name，这里统一以下划线形式登记。
TOOL_NAME_MONITOR_QUERY_ERRORS = "monitor_query_errors"

# 无工具调用时的 debug 占位值。
TOOL_TYPE_NONE = "none"

# 多意图工具请求关键词：命中时交给 LLM 做工具结果整合。
TOOL_MULTI_INTENT_KEYWORDS = ("顺便", "另外", "同时", "并且", "再", "还要")

# 计算表达式最大长度，防止过长输入造成不必要解析成本。
MAX_CALC_EXPRESSION_CHARS = 120

# 计算失败日志里表达式预览长度，避免把长输入完整写进日志。
CALC_EXPRESSION_PREVIEW_CHARS = 80

# 计算结果允许的最大绝对值，避免极端表达式产生超大数字。
MAX_CALC_ABS_VALUE = 10**12

# 计算表达式允许字符集，只支持基础四则运算和括号。
ALLOWED_CALC_CHARACTERS = "0123456789+-*/(). "

# 天气工具的演示数据。
WEATHER_BY_CITY = {
    "北京": "北京今天天气晴，气温 26°C。",
    "上海": "上海今天天气多云，气温 28°C。",
    "广州": "广州今天天气阵雨，气温 30°C。",
}

# monitor.query_errors 的演示数据：按 service 返回近 30 分钟错误摘要。
# 仅 Phase 2 workflow demo 使用；真实生产接入 APM 时会被替换掉，调用方签名
# 不变（依旧是 service -> str）。
MONITOR_QUERY_ERRORS_BY_SERVICE = {
    "payment-service": (
        "payment-service 最近 30 分钟累计 412 次 5xx，"
        "其中 87% 来自下游 bank-adapter 超时；错误集中在 POST /pay/charge。"
    ),
    "order-service": (
        "order-service 最近 30 分钟累计 23 次 5xx，无显著聚集，" "疑似偶发网络抖动。"
    ),
    "user-service": "user-service 最近 30 分钟无 5xx 错误。",
}
MONITOR_QUERY_ERRORS_DEFAULT = "未找到该 service 的监控数据（demo 数据集）。"
# 为 service 参数做基础校验：过长输入直接拒绝，避免 mock 被当作 echo 滥用。
MONITOR_SERVICE_MAX_CHARS = 64

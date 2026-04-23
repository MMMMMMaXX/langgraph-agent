"""工具调用相关常量。"""

# 天气工具名称，必须和 function calling schema 中的 name 一致。
TOOL_NAME_GET_WEATHER = "get_weather"

# 计算工具名称，必须和 function calling schema 中的 name 一致。
TOOL_NAME_CALCULATE = "calculate"

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

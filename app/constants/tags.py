"""集中管理项目内标签、标签权重和标签匹配词。"""

# 天气主题标签：用于天气工具结果、天气类记忆和检索加权。
TAG_WEATHER = "weather"

# 气候主题标签：用于稳定气候知识和城市气候类记忆。
TAG_CLIMATE = "climate"

# 无障碍概念标签：用于 accessibility/a11y 相关知识。
TAG_ACCESSIBILITY = "accessibility"

# WAI-ARIA 概念标签：用于 WAI-ARIA 相关知识。
TAG_WAI_ARIA = "wai-aria"

# 虚拟列表概念标签：用于 virtual list 相关知识。
TAG_VIRTUAL_LIST = "virtual-list"

# 当前项目支持识别的城市标签。
CITY_TAGS = ("北京", "上海", "广州", "深圳")

# 当前项目支持识别的主题标签。
TOPIC_TAGS = (TAG_WEATHER, TAG_CLIMATE)

# 当前项目支持识别的概念标签。
CONCEPT_TAGS = (TAG_ACCESSIBILITY, TAG_WAI_ARIA, TAG_VIRTUAL_LIST)

# 城市标签权重：城市命中通常强约束查询对象，权重高于普通主题词。
CITY_TAG_WEIGHTS = {city: 2.0 for city in CITY_TAGS}

# 主题标签权重：天气/气候用于辅助区分同一城市下的问题类型。
TOPIC_TAG_WEIGHTS = {
    TAG_WEATHER: 1.0,
    TAG_CLIMATE: 1.0,
}

# 概念标签权重：专有概念对文档检索有较强指向性。
CONCEPT_TAG_WEIGHTS = {
    TAG_ACCESSIBILITY: 1.5,
    TAG_WAI_ARIA: 1.5,
    TAG_VIRTUAL_LIST: 1.5,
}

# 全量标签权重：供 doc/memory keyword scoring 统一取值。
TAG_WEIGHTS = {
    **CITY_TAG_WEIGHTS,
    **TOPIC_TAG_WEIGHTS,
    **CONCEPT_TAG_WEIGHTS,
}

# 字面完全包含加权：用于短查询直接出现在内容中的补充打分。
LITERAL_MATCH_WEIGHT = 1.5

# 文档检索匹配词：范围略宽，用来增强 keyword recall。
TAG_MATCH_TERMS = {
    TAG_WEATHER: ("weather", "天气"),
    TAG_CLIMATE: ("climate", "气候"),
    TAG_ACCESSIBILITY: ("accessibility", "无障碍", "a11y"),
    TAG_WAI_ARIA: ("wai-aria", "aria", "无障碍互联网应用"),
    TAG_VIRTUAL_LIST: ("virtual-list", "虚拟列表"),
}

# 标签抽取匹配词：范围保持原有行为，避免把 doc 检索扩展词误写进 memory tags。
TAG_EXTRACTION_TERMS = {
    TAG_WEATHER: ("天气",),
    TAG_CLIMATE: ("气候",),
    TAG_ACCESSIBILITY: ("无障碍",),
    TAG_WAI_ARIA: ("WAI-ARIA", "aria"),
    TAG_VIRTUAL_LIST: ("虚拟列表",),
}

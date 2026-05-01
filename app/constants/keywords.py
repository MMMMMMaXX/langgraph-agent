"""集中管理用户意图识别、记忆分类和写入策略使用的关键词。"""

# 数学工具触发符：只有同时出现数字和这些符号时，才认为是计算请求。
MATH_OPERATOR_KEYWORDS = ("+", "-", "*", "/")

# 数学工具触发词：用于识别“算一下”这类自然语言计算请求。
MATH_QUERY_KEYWORDS = ("算",)

# 天气工具触发词：偏实时查询，通常交给 tool_agent。
WEATHER_QUERY_KEYWORDS = ("天气",)

# 知识检索触发词：偏稳定知识/概念解释，通常交给 rag_agent。
KNOWLEDGE_QUERY_KEYWORDS = (
    "气候",
    "什么",
    "是什么",
    "技术",
    "原理",
    "概念",
    "无障碍",
    "虚拟列表",
    "WAI-ARIA",
)

# 总结类触发词：用户希望聚合历史问题或对话内容。
SUMMARY_QUERY_KEYWORDS = ("总结", "回顾", "梳理", "最近所有", "列出")

# 最近/历史回忆触发词：用户希望查询刚才或之前发生过什么。
RECALL_QUERY_KEYWORDS = ("刚刚", "刚才", "上一次", "之前")

# 强时效总结触发词：只应该优先使用当前进程 Working Memory。
IMMEDIATE_SUMMARY_QUERY_KEYWORDS = ("刚才", "刚刚")

# 存在性查询触发词：用户在问是否查过、有没有问过某内容。
EXISTENCE_QUERY_KEYWORDS = ("有没有", "是否", "查过", "问过")

# 重复回答触发词：用户要求复述上一轮信息。
REPEAT_QUERY_KEYWORDS = ("再说一遍", "重复", "重新说", "复述")

# 短追问前缀：用于识别“那上海呢”这类依赖上一轮上下文的补全问题。
FOLLOWUP_QUERY_PREFIXES = ("那",)

# 短追问后缀：用于识别“那上海呢/上海呢？”这类省略主题的问题。
FOLLOWUP_QUERY_SUFFIXES = ("呢", "呢？", "呢?")

# 短追问最大长度，过长句子不按省略追问处理，避免误路由。
FOLLOWUP_QUERY_MAX_CHARS = 20

# 定义类问题信号词：同时驱动 query_classifier 的分类决策和 catalog 的 FTS 排序增益。
# 覆盖：概念类 + 操作/使用类 + 功能/作用类
# 注意：两处逻辑共享此常量，修改时需同时考虑分类行为和检索排序影响。
DEFINITION_QUERY_KEYWORDS = (
    # 概念类
    "是什么",
    "什么是",
    "定义",
    "概念",
    "含义",
    "是什么意思",
    "指的是",
    "什么叫",
    "怎么理解",
    # 操作/使用类
    "怎么用",
    "如何使用",
    "怎么使用",
    "如何配置",
    "怎么配置",
    "如何设置",
    "怎么设置",
    # 功能/作用类
    "有什么作用",
    "用来做什么",
    "有什么功能",
    "有什么用",
    "能做什么",
)

# 历史/总结类规则最大匹配长度，避免长文本正文被关键词误伤。
META_HISTORY_QUERY_MAX_CHARS = 80

# 历史/元问题总入口：supervisor、memory 节点都应该复用这组词。
META_HISTORY_QUERY_KEYWORDS = (
    *RECALL_QUERY_KEYWORDS,
    *EXISTENCE_QUERY_KEYWORDS,
    "总结",
    "回顾",
    "再说一遍",
)

# 长期向量记忆写入屏蔽词：这类问题描述的是对话过程，不是可复用事实。
VECTOR_STORE_BLOCK_KEYWORDS = (
    *RECALL_QUERY_KEYWORDS,
    *EXISTENCE_QUERY_KEYWORDS,
    "哪里",
    "总结",
    "回顾",
    *REPEAT_QUERY_KEYWORDS,
)

# 记忆分类：总结型问题。
SUMMARY_MEMORY_KEYWORDS = SUMMARY_QUERY_KEYWORDS

# 记忆分类：元问题/历史查询。
META_MEMORY_KEYWORDS = (*EXISTENCE_QUERY_KEYWORDS, *RECALL_QUERY_KEYWORDS)

# 记忆分类：用户纠错，后续可用于覆盖旧事实。
CORRECTION_MEMORY_KEYWORDS = ("不对", "错了", "纠正", "更正", "应该是", "不是")

# 记忆分类：用户偏好，后续可用于个性化回答。
PREFERENCE_MEMORY_KEYWORDS = (
    "以后回答",
    "以后请",
    "请记住",
    "记住",
    "偏好",
    "喜欢",
    "不喜欢",
    "希望你",
    "尽量",
    "风格",
)

# 记忆分类：任务状态，记录项目推进、阶段计划和下一步。
TASK_STATE_MEMORY_KEYWORDS = (
    "下一步",
    "接下来",
    "继续",
    "先做",
    "开始",
    "完成",
    "阶段",
    "计划",
    "目标",
    "我们现在",
)


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """判断文本是否包含任一关键词，统一替代散落的 any(k in text...)。"""

    return any(keyword in text for keyword in keywords)

"""运行时会话恢复来源常量。

这些常量用于标记一次会话快照是从哪里恢复出来的，方便：
1. debug / tracing 快速判断当前 turn 依赖的是 cache 还是 checkpoint
2. 避免业务代码里散落 "cache" / "checkpoint" / "empty" 这类字符串
"""

# 没有命中任何已有状态，当前 turn 从空会话开始。
RUNTIME_RESTORE_FROM_EMPTY = "empty"

# 命中了进程内 session cache，说明当前进程还保留着热状态。
RUNTIME_RESTORE_FROM_CACHE = "cache"

# 进程内 cache 没有命中，但从 LangGraph checkpoint 恢复出了会话状态。
RUNTIME_RESTORE_FROM_CHECKPOINT = "checkpoint"

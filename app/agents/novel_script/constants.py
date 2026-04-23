"""novel_script 子图的流程阈值与默认配置。"""

# 短文本以下才开启 review，避免长小说片段把“初稿 + 审查 + 返修”链路拖到过长。
REVIEW_ENABLED_MAX_SOURCE_CHARS = 4500

# 超过这个长度时降低默认场景数，控制创作链路里的 LLM 调用次数。
FAST_MODE_SOURCE_CHARS = 2500

# 长文本默认拆成 2 个场景，减少写作和 review 的模型调用次数。
FAST_MODE_TARGET_SCENE_COUNT = 2

# 短文本默认拆成 3 个场景，保证改编结构足够完整。
DEFAULT_TARGET_SCENE_COUNT = 3

# 每个被 review 标记的问题场景最多返修几次，防止 review / rewrite 无限循环。
MAX_SCENE_REWRITE_ATTEMPTS = 1

# split_into_scenes 与 extract_story_facts 这两个前置动作的固定预算。
NOVEL_SCRIPT_SETUP_ITERATIONS = 2

# 首轮审查与返修后复审各需要一次动作预算。
NOVEL_SCRIPT_REVIEW_CYCLE_ITERATIONS = 2

# 不开启 review 时，额外保留一轮 finalize 兜底空间。
NOVEL_SCRIPT_NO_REVIEW_FINALIZE_BUFFER = 1

# story facts 抽取前的原文截断长度，避免事实抽取阶段吃完整长文。
FACT_SOURCE_TEXT_MAX_CHARS = 1800

# planner 只需要很短的原文预览，用于识别当前任务和保持上下文轻量。
PLANNER_SOURCE_PREVIEW_CHARS = 240

# planner 中最多透出的故事人物数量。
PLANNER_FACT_CHARACTER_LIMIT = 8

# planner 中最多透出的故事地点数量。
PLANNER_FACT_LOCATION_LIMIT = 6

# planner 中最多透出的故事目标数量。
PLANNER_FACT_GOAL_LIMIT = 6

# planner 中最多透出的故事冲突数量。
PLANNER_FACT_CONFLICT_LIMIT = 4

# planner 中单个 scene summary 的预览长度。
PLANNER_SCENE_SUMMARY_PREVIEW_CHARS = 80

# planner 中单个 scene draft 的预览长度。
PLANNER_SCENE_DRAFT_PREVIEW_CHARS = 100

# planner 只透出最近几条 review issue，避免 planner 上下文膨胀。
PLANNER_REVIEW_ISSUE_LIMIT = 3

# planner 中单条 review issue 的预览长度。
PLANNER_REVIEW_ISSUE_PREVIEW_CHARS = 80

# planner 中单条 review target reason 的预览长度。
PLANNER_REVIEW_TARGET_REASON_PREVIEW_CHARS = 60

# planner 中最多展示的 pending rewrite scene 数。
PLANNER_PENDING_REWRITE_PREVIEW_LIMIT = 5

# planner 中最多展示的 scene plan / scene draft 条数。
PLANNER_SCENE_PREVIEW_LIMIT = 6

# review 阶段原文摘要截断长度。
REVIEW_SOURCE_PREVIEW_CHARS = 600

# review 阶段单个场景草稿预览长度。
REVIEW_SCENE_DRAFT_PREVIEW_CHARS = 500

# review prompt 中各类事实最多保留的条数。
REVIEW_FACT_CHARACTER_LIMIT = 10
REVIEW_FACT_LOCATION_LIMIT = 8
REVIEW_FACT_GOAL_LIMIT = 8
REVIEW_FACT_CONFLICT_LIMIT = 6

# review 模型输出 token 上限，审查只需要结构化问题列表，不需要长篇解释。
REVIEW_MAX_COMPLETION_TOKENS = 140

# 单场景剧本生成 token 上限。
WRITE_SCENE_MAX_COMPLETION_TOKENS = 320

# 场景切分时摘要字段的截断长度。
SCENE_SUMMARY_MAX_CHARS = 80

# 最终改编说明最多展示多少条审查重点。
FINAL_REVIEW_NOTE_LIMIT = 3

# tool history 中工具输出预览长度。
TOOL_OUTPUT_PREVIEW_CHARS = 200

# observation 字段预览长度。
OBSERVATION_PREVIEW_CHARS = 300

# planner thought 调试预览长度。
PLANNER_THOUGHT_PREVIEW_CHARS = 80

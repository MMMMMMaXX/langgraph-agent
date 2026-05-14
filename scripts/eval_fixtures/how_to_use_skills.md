Skills 构建指南与实践

本次分享主要内容，来自 Anthropic 上个月发布的 Claude Skills 权威构建指南，和大家一起分享学习一下个人的收获和实践。

[附件]

# Skills 构建指南

## 基础知识

#### \*\* \*\*Skills = 可被 AI 自动调用的“能力插件”，让 AI 从“会对话”升级为“能干活”

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=084db177c2464d9ca313e88aa58fa3ff&docGuid=k4dUtMEyV3zOro)

「Skills」的概念最先由 Anthropic 公司提出

1. Agent Skills 是用于扩展 AI 助手能力的 **模块化能力单元\*\***。\*\*
2. 每个 Skills 由一份 **SKILL.md 说明文件** 以及可选的 **脚本与模板** 组成，用于定义其使用方式与执行逻辑。
3. Skills 采用 **模型自动调用机制**：AI 会基于当前上下文自动判断是否需要调用对应 Skills，无需用户显式干预，从而实现**更智能、更自然的任务执行体验**。

### 三个核心设计理念

**1. 渐进式披露（Progressive Disclosure）**

技能用三层结构来组织内容：

- • **第一层（YAML 前置信息）**：每次都会加载到 Claude 的系统提示里。内容要精简——让 Claude 知道"这个技能是做什么的、什么时候该用它"就够了，不需要把所有东西都塞进来。
- • **第二层（SKILL.md 正文）**：当 Claude 觉得当前任务跟这个技能有关，才会加载完整指令。
- • **第三层（链接文件）**：放在技能文件夹里的其他文件，Claude 可以按需查阅。

这种分层设计的好处：既省 token，又能保留足够的专业深度。

**2. 可组合性（Composability）**

Claude 可以同时加载多个技能。你的技能要能跟其他技能和平共处，别假设自己是唯一被启用的那个。

**3. 可移植性（Portability）**

在 Claude.ai、Claude Code 和 API 里，技能的工作方式完全一样。写一次，到处能用——前提是运行环境支持技能所需的依赖。

#### 1.为什么需要 Skills

#### ——让 Agent 获取系统化「技能」，告别重复劳动&低效协作

**高价值的方法、经验与最佳实践，未被系统化沉淀，导致大量重复劳动和低效协作。**

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=7f10d9fde4ec4222954c21967cef90cd&docGuid=k4dUtMEyV3zOro)
Skills 为解决这一问题而生：**把零散的 AI 使用方式，升级为结构化、可复用的工作能力，让 AI 更懂你的 SOP。**

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=2a2ebeefcbd543828ef62130e720eb58&docGuid=k4dUtMEyV3zOro)

#### 2.Skills 能带来什么

#### ——提效！提效！提效！

##### **让「个人」减少重复劳动**

**显著缩短任务完成时间，让精力聚焦在更高价值的思考与决策上。**

- 将高频、重复的工作流程进行标准化和模板化，减少从零开始
- 将复杂任务拆解为可直接调用的步骤，降低使用门槛
- 将零散经验沉淀为稳定产出，避免反复试错

##### **为「团队」沉淀能力资产**

**经验可复用、能力可复制、团队整体提效，推动团队从“个体作战”走向“体系化协作”。**

- 将团队通用方法论沉淀为 Skills，形成统一工作范式
- 将成熟流程产品化，降低新人成长与协作成本
- 将专家经验结构化，放大个人价值，形成规模化能力供给

#### 3.谁会需要 Skills

#### ——高效党、有想法、强实践的你

**如果你：**

- 不想每天从零开始、重复无效劳动
- 想从执行中解放出来，聚焦思考和决策
- 希望 AI 不仅会对话，更能帮你干好活

**👉 \*\***强烈推荐使用 Skills，让 AI 进化为你的专属业务助手\*\*

**如果你：**

- 总结过高效 Prompt
- 沉淀过工作模版
- 有成熟的工作方法论

👉 **非常适合创建 Skills，把个人能力产品化。**

**如果你们：**

- 团队形成了标准化的业务流程
- 沉淀了跨岗位协作的 SOP
- 积累了某个领域的专业最佳实践

**👉 \*\***不妨创建并分享 Skills，沉淀可量化的团队资产。\*\*

参考链接：[在度厂，这样玩转 Skills！](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/z97pwhZizu/hy296wAZ7z/wWWa0TjmqqgiWR?track=yonghuqun?t=mention&mt=doc&dt=doc)

---

## 规划与设计

### 第一步：从用例出发

在写任何指令前，先想清楚这个技能要解决的具体场景

一个好的用例定义长这样：

```bash
用例：项目冲刺规划

触发条件：用户说"帮我规划这个冲刺"或"创建冲刺任务"

步骤：
  1. 通过 MCP 获取 Linear 的当前项目状态
  2. 分析团队速度和容量
  3. 建议任务优先级
  4. 在 Linear 里创建带标签和估算的任务

结果：冲刺计划规划完毕，任务全部创建好
```

问自己这几个问题：

- 用户想达成什么目标？
- 这需要哪些多步骤的工作流程？
- 需要哪些工具（Claude 内置的，还是 MCP 的）？
- 需要嵌入哪些领域知识或最佳实践？

#### 常见用例分析

Anthropic 观察到三种最常见的技能使用场景：

**第 1 类：文档和素材创建**

适合场景：创建一致的高质量输出，比如文档、演示文稿、应用、设计、代码等。

真实案例：`frontend-design` 技能（另见 docx、pptx、xlsx 相关技能）

```
"创建有辨识度的、生产级别的前端界面，注重设计质量。在构建 Web 组件、页面、作品、海报或应用时使用。"
```

关键做法：

- **嵌入风格指南**和品牌规范
- **用模板结构保证输出一致**
- **最终确认前做质量清单检查**
- 不需要外部工具——用 Claude 的内置能力就够

**第 2 类：工作流程自动化**

适合场景：多步骤流程，需要一致的执行方式，可能跨多个 MCP 服务器协调。

真实案例：`skill-creator` 技能

```
"创建新技能的交互式向导。引导用户完成用例定义、前置信息生成、指令撰写和验证。"
```

关键做法：

- **带验证节点的分步工作流**
- **常见结构的模板**
- **内置审查和改进建议**
- **迭代优化循环**

**第 3 类：MCP 增强**

适合场景：为 MCP 服务器的工具访问能力加上工作流程引导。

真实案例：来自 Sentry 的 `sentry-code-review` 技能

```
"利用 Sentry 错误监控数据，通过其 MCP 服务器自动分析和修复 GitHub Pull Requests 中发现的 bug。"
```

关键做法：

- 按顺序协调多个 MCP 调用
- **嵌入领域专业知识**
- **提供用户本来需要手动指定的上下文**
- 处理常见 MCP 问题

### 定义成功标准

怎么知道你的技能跑通了？

先说清楚：下面这些是参考目标，不是死板的门槛——测试时还是需要一定的感性判断。Anthropic 正在开发更完善的量化评估工具。

**量化指标：**

- 技能在 90% 的相关查询中触发

  - 怎么测：跑 10-20 个应该触发技能的测试问题，记录自动加载的次数

- 工作流程在 X 次工具调用内完成

  - 怎么测：有无技能各跑一次相同任务，对比工具调用次数和 token 消耗

- • 每个工作流程 0 次 API 调用失败
  - 怎么测：测试期间监控 MCP 服务器日志，追踪重试率和错误码

**定性指标：**

- 用户不需要提示 Claude 下一步该做什么
- 工作流程无需用户纠正就能跑完
- 跨会话结果保持一致

### 技术要求

```bash
your-skill-name/
├── SKILL.md              # 必须——主技能文件
├── scripts/              # 可选——可执行代码
│   ├── process_data.py
│   └── validate.sh
├── references/           # 可选——参考文档
│   ├── api-guide.md
│   └── examples/
└── assets/               # 可选——模板等资源
    └── report-template.md
```

### 几条必须遵守的规则

**SKILL.md 命名：**

- 必须精确写成 `SKILL.md`（区分大小写）
- 任何变体都不行：`SKILL.MD`、`skill.md` 都不能用

**技能文件夹命名：**

- ✅ 用 kebab-case（短横线连接小写）：`notion-project-setup`
- ❌ 不能有空格：`Notion Project Setup`
- ❌ 不能用下划线：`notion_project_setup`
- ❌ 不能有大写：`NotionProjectSetup`

**不要放 README.md：**

- 技能文件夹内别放 README.md
- 所有文档放在 SKILL.md 或 references/ 里

### YAML 前置信息：最关键的部分

YAML 前置信息是 Claude 决定要不要加载你的技能的依据，务必写对。

```bash
---
name: your-skill-name
description: 它干什么。当用户说[具体短语]时使用。
---
```

#### 各字段说明

`name`**（必填）：**

- 只能用 kebab-case
- 没有空格，没有大写
- 最好和文件夹名一致

`description`**（必填）：**

- 必须同时包含：

  - 这个技能能做什么
  - 什么时候该用（触发条件）

- 最多 1024 个字符
- 不能含 XML 标签（`<` 或 `>`）
- 要包含用户可能实际说出的话
- 如果涉及特定文件类型，要提到

**安全限制：**

- 禁止用 XML 尖括号（`<` `>`）
- 技能名称不能以 "claude" 或 "anthropic" 开头（保留词）
- 原因：前置信息会进入 Claude 的系统提示，恶意内容可能被用来注入指令

### 写好 description 字段

Anthropic 工程博客说过："这段元数据……提供了恰好足够让 Claude 知道该用哪个技能的信息，而不需要把所有内容都加载到上下文里。"这就是渐进式披露的第一层。

**格式模板：**`[能做什么] + [什么时候用] + [主要功能]`\*\*\*\*

```
# 好——具体且可操作
description: 分析 Figma 设计文件并生成开发者交接文档。
  当用户上传 .fig 文件，或者问"设计规格"、"组件文档"、
  "设计转代码交接"时使用。

# 好——包含触发词
description: 管理 Linear 项目工作流，包括冲刺规划、任务创建和状态追踪。
  当用户提到"冲刺"、"Linear 任务"、"项目规划"，
  或者要求"创建工单"时使用。

# 好——价值主张清晰
description: PayFlow 的端到端客户引导工作流。
  处理账户创建、支付设置和订阅管理。
  当用户说"引导新客户"、"设置订阅"或"创建 PayFlow 账户"时使用。
```

```
# 太模糊
description: 帮助处理项目。

# 没有触发条件
description: 创建复杂的多页文档系统。

# 太技术化，用户不会这么说
description: 实现具有层次关系的 Project 实体模型。

```

### 写主体指令

过了 YAML 前置信息，就用 Markdown 写具体的指令。

**推荐结构：**

````md
---
name: your-skill
description: [...]
---

# 你的技能名称

## 指令

### 第 1 步：[第一个主要步骤]

清楚说明这步要做什么。

示例：

```bash
python scripts/fetch_data.py --project-id PROJECT_ID
预期输出：[描述成功时应该看到什么]
```
````

### 第 2 步：[下一步骤，按需添加]

## 示例

### 示例 1：[常见场景]

用户说："设置一个新的营销活动"
操作：

1. 通过 MCP 获取现有活动
2. 用提供的参数创建新活动
   结果：活动创建成功，附上确认链接

## 故障排查

错误：[常见错误消息]
原因：[为什么会出现]
解决方法：[怎么修]

```


**写指令的最佳实践：**

**✅ 要具体，要说清楚怎么操作：**

```

运行 `python scripts/validate.py --input {filename}` 来检查数据格式。
如果验证失败，常见原因有：

- 缺少必填字段（把它加到 CSV 里）
- 日期格式不对（要用 YYYY-MM-DD）

```
**❌ 别含糊其词：**

```

继续之前请先验证数据。

```
**✅ 要包含错误处理：**

```

## 常见问题

### MCP 连接失败

看到 "Connection refused" 时：

1. 确认 MCP 服务器在运行：检查 设置 > 扩展
2. 确认 API 密钥有效
3. 尝试重连：设置 > 扩展 > [你的服务] > 重新连接

```
**✅ 要用渐进式引用：**

```

写查询之前，请先看 `references/api-patterns.md`，里面有：

- 速率限制指南
- 分页模式
- 错误码和处理方式

```
**SKILL.md 要保持聚焦**：核心指令放在 SKILL.md，详细文档移到 `references/` 里，通过链接引用。



## 测试与迭代
💡 **专业技巧：先把一个难题跑通，再推广**

我们发现，最高效的做法是：先针对一个最具挑战的任务反复迭代，直到 Claude 成功搞定，再把这个成功方案提炼成技能。这样能充分利用 Claude 的上下文学习能力，比广撒网测试更快得到有效反馈。有了可运行的基础后，再扩展到多个测试用例来验证覆盖面。

### 推荐测试方法
#### 1、触发测试
目标：确保技能在对的时机加载。

```

应该触发：

- "帮我设置一个新的 ProjectHub 工作区"
- "我需要在 ProjectHub 里创建个项目"
- "为 Q4 规划初始化一个 ProjectHub 项目"

不应该触发：

- "旧金山今天天气怎么样？"
- "帮我写个 Python 脚本"
- "创建一个电子表格"（除非你的技能支持表格）

```
#### **2. 功能测试**
目标：验证技能输出是对的。

```

测试：创建包含 5 个任务的项目
给定：项目名 "Q4 规划"，5 个任务描述
执行：技能运行工作流
验证：

- ProjectHub 里创建了项目
- 5 个任务属性都正确
- 所有任务都关联到项目
- 没有 API 报错

```
#### 3、对比测试
目标：证明技能比没有技能时更好。

|指标|无技能|有技能|
|-|-|-|
|用户如何提供指令|每次都要|自动执行|
|来回消息数|15 条|2 个澄清问题|
|API 调用失败次数|3 次（需要重试）|0 次|
|消耗的 token|12,000|6,000|

### 用 skill-creator 工具
**创建技能：**

* • 从自然语言描述生成技能
* • 自动产生格式正确的 SKILL.md
* • 建议触发词和结构

**审查技能：**

* • 标出常见问题：描述太模糊、缺少触发条件、结构有问题
* • 识别过度/不足触发的风险
* • 根据技能目的建议测试用例

**迭代改进：**

* • 遇到边缘案例或失败后，把问题带回 skill-creator，让它帮你改进

使用方法：

```

"使用 skill-creator 帮我为[你的用例]构建一个技能"

```
> ⚠️ 注意：skill-creator 帮你设计和打磨技能，但不会跑自动化测试套件，也不会生成量化评估结果。
### 根据反馈持续迭代
技能是"活文档"，要根据实际使用情况不断调整。

**触发不足的信号：**

* • 技能该自动加载时没加载
* • 用户手动开启它
* • 有人问"什么时候用这个技能"

→ 解决办法：在 description 里加更多细节和关键词（包括专业术语）

**触发过度的信号：**

* • 技能在不相关的查询里也触发了
* • 用户禁用了它
* • 用户搞不清它的用途

→ 解决办法：添加负触发词，让描述更具体

**执行问题的信号：**

* • 结果不一致
* • API 调用失败
* • 需要用户帮忙纠正

→ 解决办法：改进指令，加上错误处理

## 模式和故障排查
### 模式一：顺序工作流编排
**适合场景：** 用户需要按特定顺序执行多个步骤。

关键技术：步骤顺序要明确、步骤间依赖关系要写清楚、每个阶段要有验证、失败时要有回滚指令。

### 模式二：多 MCP 协调
**适合场景****：** 工作流需要跨多个服务

关键技术：阶段分隔要清晰、MCP 间的数据传递要明确、进下一阶段前要验证、错误处理要集中。

### 模式三：迭代式优化
**适合场景：** 输出质量需要通过多轮迭代来提升。

关键技术：质量标准要明确、迭代改进要有节奏、验证脚本要好用、要知道什么时候停。

*💡 ***进阶技巧***：对于关键验证步骤，可以考虑打包一个脚本来做程序化检查，而不是靠自然语言描述。代码的执行是确定的，语言的理解不是。参考 Office 技能系列的示例。*

### 模式四：情境感知工具选择
**适合场景：** 同一个目标，根据上下文要选不同的工具。

关键技术：决策标准要清晰、要有备用选项、选择理由要透明。

### 模式五：特定领域智能
**适合场景：** 你的技能在工具访问之外还需要提供专业知识。

关键技术：领域专业知识要嵌入逻辑、行动前要合规检查、记录要完整、治理边界要清晰。

### 故障排查
#### **技能不触发**
症状：技能从来不自动加载

解决：修改 description 字段（参考前面的好/坏示例）

自查清单：

* • 描述是不是太笼统？（"帮助处理项目"不够用）
* • 有没有用户可能实际说的触发词？
* • 如果涉及文件类型，有没有提到？

调试技巧：问 Claude："你什么时候会用 [技能名] 这个技能？"Claude 会引用 description 里的内容。根据缺失的部分来调整。

#### **技能触发太频繁**
症状：技能在不相关的查询里也出现了

解决方法：

1、加负触发词：

```

description: 用于 CSV 文件的高级数据分析，适合统计建模、回归、聚类分析。
不要用于简单数据探索（请改用 data-viz 技能）。

```
2、描述更具体：

```

# 太宽泛

description: 处理文档

# 更好

description: 处理 PDF 法律文件以供合同审查

```
3、明确使用范围：

```

description: PayFlow 电子商务支付处理。专门用于在线支付工作流，
不适用于一般财务查询。

```
#### **指令没有被遵守**
症状：技能加载了，但 Claude 没按指令执行

常见原因及解决方法：

**1、指令太冗长**

    * • 保持简洁
    * • 多用列表和编号
    * • 详细参考文档移到单独文件

**2、关键指令被埋没**

    * • 重要内容放在最前面
    * • 用 `## 重要` 或 `## 关键` 这样的标题
    * • 关键点可以重复

**3、语言模糊**

```

# 不好

确保正确验证相关内容

# 好

关键：在调用 create_project 之前，必须确认：

- 项目名称非空
- 至少分配了一个团队成员
- 开始日期不在过去

```
> 💡 **进阶技巧**：对于关键验证步骤，考虑打包脚本来程序化执行检查，而不是依赖自然语言指令。代码执行是确定的，语言理解不是。参考 Office 技能系列的示例。
**4、模型"偷懒"**——加上明确的激励说明：

```

## 执行说明

- 请花足够时间彻底完成
- 质量比速度更重要
- 不要跳过验证步骤

```
注意：这段加在用户提示里比加在 SKILL.md 里效果更好。

#### **上下文太大导致响应变慢或质量下降**
原因：

* • SKILL.md 内容太多
* • 同时启用的技能太多
* • 所有内容都加载了，没有用渐进式披露

解决方法：

1. 1. 精简 SKILL.md

    * • 详细文档移到 `references/`
    * • 用链接引用而非内联
    * • SKILL.md 尽量控制在 5,000 个词以内

2. 2. 减少同时启用的技能数量

    * • 超过 20-50 个同时启用的技能就要考虑精简了
    * • 建议按需选择性启用
    * • 相关功能可以打包成技能"组合包"

### 可供参考的官方文档
**Anthropic 官方资源：**

* • Best Practices Guide（最佳实践指南）
* • Skills Documentation（技能文档）
* • API Reference（API 参考）
* • MCP Documentation（MCP 文档）

**官方博客文章：**

* • Introducing Agent Skills
* • Engineering Blog: Equipping Agents for the Real World
* • Skills Explained
* • How to Create Skills for Claude
* • Building Skills for Claude Code
* • Improving Frontend Design through Skills



## `skill-creator`技能泛读
上面啰嗦了这么多理论，有些枯燥，正好Anthropic最近刚更新了`skill-creator`技能，下面一块来看下官方的`skill-creator`是怎么写的。

```

---

name: skill-creator
description: 【描述功能】Create new skills, modify and improve existing skills, and measure skill performance. 【调用时机】Use when users want to create a skill from scratch, edit, or optimize an existing skill, run evals to test a skill, benchmark skill performance with variance analysis, or optimize a skill's description for better triggering accuracy.

---

# Skill Creator

A skill for creating new skills and iteratively improving them.

【从大的方向介绍整体的工作流程，分条目列出步骤】
At a high level, the process of creating a skill goes like this:

- Decide what you want the skill to do and roughly how it should do it
- Write a draft of the skill
- Create a few test prompts and run claude-with-access-to-the-skill on them
- Help the user evaluate the results both qualitatively and quantitatively
  - While the runs happen in the background, draft some quantitative evals if there aren't any (if there are some, you can either use as is or modify if you feel something needs to change about them). Then explain them to the user (or if they already existed, explain the ones that already exist)
  - Use the `eval-viewer/generate_review.py` script to show the user the results for them to look at, and also let them look at the quantitative metrics
- Rewrite the skill based on feedback from the user's evaluation of the results (and also if there are any glaring flaws that become apparent from the quantitative benchmarks)
- Repeat until you're satisfied
- Expand the test set and try again at larger scale

补充说明，略

【向模型解释为什么要这么做，后面会提到原因】

## Communicating with the user

技能创建者可能会被各种对编程术语熟悉程度不同的人使用。如果你还没听说过（你怎么可能没听说过，它才刚刚兴起），现在有一种趋势是，Claude 的强大功能正在激励水管工打开他们的终端，父母和祖父母们开始在谷歌上搜索“如何安装 npm”。另一方面，大多数用户可能都具备相当的计算机素养。
所以，请留意上下文线索，以理解如何措辞你的交流！

---

## Creating a skill

### Capture Intent

获取用户意图，解答下面的问题
【通过问题明确新建一个 skills 的各个要素，在自己新建 skills 的时候，也需要回答这些问题】

1. What should this skill enable Claude to do?
2. When should this skill trigger? (what user phrases/contexts)
3. What's the expected output format?
4. Should we set up test cases to verify the skill works? Skills with objectively verifiable outputs (file transforms, data extraction, code generation, fixed workflow steps) benefit from test cases. Skills with subjective outputs (writing style, art) often don't need them. Suggest the appropriate default based on the skill type, but let the user decide.

### Interview and Research

略

### Write the SKILL.md

Based on the user interview, fill in these components:

按照 skills 的固定格式生成
【明确 skills 的固定格式】

- **name**: Skill identifier
- **description**: When to trigger, what it does. This is the primary triggering mechanism - include both what the skill does AND specific contexts for when to use it. All "when to use" info goes here, not in the body. Note: currently Claude has a tendency to "undertrigger" skills -- to not use them when they'd be useful. To combat this, please make the skill descriptions a little bit "pushy". So for instance, instead of "How to build a simple fast dashboard to display internal Anthropic data.", you might write "How to build a simple fast dashboard to display internal Anthropic data. Make sure to use this skill whenever the user mentions dashboards, data visualization, internal metrics, or wants to display any kind of company data, even if they don't explicitly ask for a 'dashboard.'"
- **compatibility**: Required tools, dependencies (optional, rarely needed)
- **the rest of the skill :)**

### Skill Writing Guide

【给出结构的示例】

#### Anatomy of a Skill

```
skill-name/
├── SKILL.md (required)
│   ├── YAML frontmatter (name, description required)
│   └── Markdown instructions
└── Bundled Resources (optional)
    ├── scripts/    - Executable code for deterministic/repetitive tasks
    ├── references/ - Docs loaded into context as needed
    └── assets/     - Files used in output (templates, icons, fonts)
```

#### Progressive Disclosure

【最关键的渐进式暴露】
Skills use a three-level loading system:

1. **Metadata** (name + description) - Always in context (~100 words)
2. **SKILL.md body** - In context whenever skill triggers (<500 lines ideal)
3. **Bundled resources** - As needed (unlimited, scripts can execute without loading)

These word counts are approximate and you can feel free to go longer if needed.

【核心范式】
**Key patterns:**

- Keep SKILL.md under 500 lines; if you're approaching this limit, add an additional layer of hierarchy along with clear pointers about where the model using the skill should go next to follow up.
- Reference files clearly from SKILL.md with guidance on when to read them
- For large reference files (>300 lines), include a table of contents

【可以通过指定不同的文件来适配不同的平台】
**Domain organization**: When a skill supports multiple domains/frameworks, organize by variant:

```
cloud-deploy/
├── SKILL.md (workflow + selection)
└── references/
    ├── aws.md
    ├── gcp.md
    └── azure.md
```

Claude reads only the relevant reference file.

#### Principle of Lack of Surprise

This goes without saying, but skills must not contain malware, exploit code, or any content that could compromise system security. A skill's contents should not surprise the user in their intent if described. Don't go along with requests to create misleading skills or skills designed to facilitate unauthorized access, data exfiltration, or other malicious activities. Things like a "roleplay as an XYZ" are OK though.

【写作范式】

#### Writing Patterns

在指令中请使用祈使句。

【祈使句（Imperative Sentence）用于表达命令、请求、劝告、警告或禁止。其核心特征是通常省略第二人称主语“你”（You），以动词原形开头，句末使用句号或感叹号。中文常带“请”、“别”、“不”】

**定义输出格式** - 你可以这样做：

```markdown
    ## 报告结构
    始终使用这个确切的模板：
    # [标题]
    ## 执行摘要
    ## 主要发现
    ## 建议
```

【给出具体的示例会很有帮助】
**示例模式**

- 包含示例会很有帮助。示例的格式可以如下（但如果示例中包含“输入”和“输出”，你可能需要稍作调整）：

# 原文：

```markdown
    ## 提交消息格式
    **示例1：**
    输入：已使用JWT令牌添加用户身份验证
    输出：feat(auth)：实现基于JWT的身份验证
```

### 写作风格

【解释原因的目的是让模型能理解底层的逻辑，而不是简单的遵循固定的命令】
尝试向模型解释事物为何重要，而不是生硬地列举陈腐的“必须做的事”。运用心理理论，努力使技能具有普遍性，而非仅限于特定示例。先写一份草稿，然后以全新的视角审视并加以改进。

### Test Cases

【使用实际可能会使用的触发语句来测试，避免离了本人就跑不起来】
After writing the skill draft, come up with 2-3 realistic test prompts — the kind of thing a real user would actually say. Share them with the user: [you don't have to use this exact language] "Here are a few test cases I'd like to try. Do these look right, or do you want to add more?" Then run them.

## Running and evaluating test cases

具体测试反馈部分，略

## Improving the skill

This is the heart of the loop. You've run the test cases, the user has reviewed the results, and now you need to make the skill better based on their feedback.

### How to think about improvements

【收集真实的反馈来改进】

1. **根据反馈进行概括** The big picture thing that's happening here is that we're trying to create skills that can be used a million times (maybe literally, maybe even more who knows) across many different prompts. Here you and the user are iterating on only a few examples over and over again because it helps move faster. The user knows these examples in and out and it's quick for them to assess new outputs. But if the skill you and the user are codeveloping works only for those examples, it's useless. Rather than put in fiddly overfitty changes, or oppressively constrictive MUSTs, if there's some stubborn issue, you might try branching out and using different metaphors, or recommending different patterns of working. It's relatively cheap to try and maybe you'll land on something great.

【可以查看 skills 执行过程中的思考过程，如果绕来绕去，即使最后达成了目标，也需要进行优化】 2. **保持 prompt 简洁** Remove things that aren't pulling their weight. Make sure to read the transcripts, not just the final outputs — if it looks like the skill is making the model waste a bunch of time doing things that are unproductive, you can try getting rid of the parts of the skill that are making it do that and seeing what happens.

【今天的大模型很聪明，有一定的心智，尝试让模型真正的理解任务，比死板的命令会有更好的效果】 3. **解释原因** Try hard to explain the **why** behind everything you're asking the model to do. Today's LLMs are _smart_. They have good theory of mind and when given a good harness can go beyond rote instructions and really make things happen. Even if the feedback from the user is terse or frustrated, try to actually understand the task and why the user is writing what they wrote, and what they actually wrote, and then transmit this understanding into the instructions. If you find yourself writing ALWAYS or NEVER in all caps, or using super rigid structures, that's a yellow flag — if possible, reframe and explain the reasoning so that the model understands why the thing you're asking for is important. That's a more humane, powerful, and effective approach.

【能通过脚本做的，就没必要再让大模型去决策，固定的流程和步骤，尽量让大模型只关心调用即可】 4. **寻找测试用例中重复出现的工作** Read the transcripts from the test runs and notice if the subagents all independently wrote similar helper scripts or took the same multi-step approach to something. If all 3 test cases resulted in the subagent writing a `create_docx.py` or a `build_chart.py`, that's a strong signal the skill should bundle that script. Write it once, put it in `scripts/`, and tell the skill to use it. This saves every future invocation from reinventing the wheel.

This task is pretty important (we are trying to create billions a year in economic value here!) and your thinking time is not the blocker; take your time and really mull things over. I'd suggest writing a draft revision and then looking at it anew and making improvements. Really do your best to get into the head of the user and understand what they want and need.

### The iteration loop

【优化完一轮后，再重复测试，直到所有的问题都被解决】
After improving the skill:

1. Apply your improvements to the skill
2. Rerun all test cases into a new `iteration-<N+1>/` directory, including baseline runs. If you're creating a new skill, the baseline is always `without_skill` (no skill) — that stays the same across iterations. If you're improving an existing skill, use your judgment on what makes sense as the baseline: the original version the user came in with, or the previous iteration.
3. Launch the reviewer with `--previous-workspace` pointing at the previous iteration
4. Wait for the user to review and tell you they're done
5. Read the new feedback, improve again, repeat

Keep going until:

- The user says they're happy
- The feedback is all empty (everything looks good)
- You're not making meaningful progress

---

【高级选项：可以通过 subagent 的形式来并行测试不同版本的 skills，从而评估改动的有效性】

## Advanced: Blind comparison

For situations where you want a more rigorous comparison between two versions of a skill (e.g., the user asks "is the new version actually better?"), there's a blind comparison system. Read `agents/comparator.md` and `agents/analyzer.md` for the details. The basic idea is: give two outputs to an independent agent without telling it which is which, and let it judge quality. Then analyze why the winner won.

This is optional, requires subagents, and most users won't need it. The human review loop is usually sufficient.

---

## Description Optimization

【描述词很关键，如果新技能的命中率比较低，或者不符合预期，主要可以通过优化描述词的形式来优化】
The description field in SKILL.md frontmatter is the primary mechanism that determines whether Claude invokes a skill. After creating or improving a skill, offer to optimize the description for better triggering accuracy.

### Step 1: Generate trigger eval queries

Create 20 eval queries — a mix of should-trigger and should-not-trigger. Save as JSON:
【可以设置一些真实可能会输入的提示词，先判定好是否应该触发该技能，用于量化的评估命中率】

```json
[
  { "query": "the user prompt", "should_trigger": true },
  { "query": "another prompt", "should_trigger": false }
]
```

The queries must be realistic and something a Claude Code or Claude.ai user would actually type. Not abstract requests, but requests that are concrete and specific and have a good amount of detail. For instance, file paths, personal context about the user's job or situation, column names and values, company names, URLs. A little bit of backstory. Some might be in lowercase or contain abbreviations or typos or casual speech. Use a mix of different lengths, and focus on edge cases rather than making them clear-cut (the user will get a chance to sign off on them).
【错误的提问方式，太过于笼统】
Bad: `"Format this data"`, `"Extract text from PDF"`, `"Create a chart"`
【正确的提问方式，包含大量明确的细节】
Good: `"ok so my boss just sent me this xlsx file (its in my downloads, called something like 'Q4 sales final FINAL v2.xlsx') and she wants me to add a column that shows the profit margin as a percentage. The revenue is in column C and costs are in column D i think"`

For the **should-trigger** queries (8-10), think about coverage. You want different phrasings of the same intent — some formal, some casual. Include cases where the user doesn't explicitly name the skill or file type but clearly needs it. Throw in some uncommon use cases and cases where this skill competes with another but should win.

For the **should-not-trigger** queries (8-10), the most valuable ones are the near-misses — queries that share keywords or concepts with the skill but actually need something different. Think adjacent domains, ambiguous phrasing where a naive keyword match would trigger but shouldn't, and cases where the query touches on something the skill does but in a context where another tool is more appropriate.
【测试负面例子的时候，要有挑战性一些，越是模棱两可的例子越有价值】
The key thing to avoid: don't make should-not-trigger queries obviously irrelevant. "Write a fibonacci function" as a negative test for a PDF skill is too easy — it doesn't test anything. The negative cases should be genuinely tricky.

### Step 2: Review with user

略

### Step 3: Run the optimization loop

略

### 技能触发机制的工作原理

了解触发机制有助于设计更好的评估查询。技能会以名称+描述的形式出现在 Claude 的`available_skills`列表中，Claude 会根据该描述决定是否调用技能。需要了解的重要一点是，Claude 仅会调用那些它无法自行轻松处理的任务的技能——像“阅读此 PDF”这样的简单、一步式查询，即使描述完全匹配，也可能不会触发技能，因为 Claude 可以直接使用基本工具处理这些查询。当描述匹配时，复杂、多步或专门的查询会可靠地触发技能。

### Step 4: Apply the result

略

Good luck!

````
补充链接：[Claude悄悄更新了Skills生成器，这绝对是一次史诗级升级。](https://mp.weixin.qq.com/s/vjMG8i7DwQ7R2B1C4AVQdA?t=mention&mt=doc&dt=sdk)

# CR Skills迭代
看完官方的示例，再来看下实际的应用：以CR Skills为例

优化前

![目录结构](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=f955b8c1d982480fbac19b0ea74f99ef&docGuid=otHaU6vfY6QfcR "目录结构")
skills.md

---

## name: code_review
description: 当用户需要代码评审、code review、代码审查、CR的时候，使用这个技能。
# Code_review
## Overview
本指南涵盖了使用sh脚本和命令行工具对本地未提交的commit做代码审查的操作。

### Code Review Workflow
* 1、**获取指令脚本**: 在项目根目录执行指令`mkdir -p .comate/skills/code_review/scripts/temp && [ -d .comate/skills/code_review/scripts/temp ] && curl http://10.25.72.105:8081/prompt/v2/talos > .comate/skills/code_review/scripts/prompt.md && curl http://10.25.72.105:8081/prompt/v2/scripts/talos > .comate/skills/code_review/scripts/talos_process.sh && chmod +x .comate/skills/code_review/scripts/talos_process.sh && .comate/skills/code_review/scripts/talos_process.sh`
* 2、**执行指令**: 上一步执行完毕后再开始按照.comate/skills/code_review/scripts/prompt.md中的prompt继续执行
* 3、**代码审查范围**: 仅针对本地未提交的commit的diff范围进行代码审查,不包括已push的代码和未commit的代码。

## Resources
### scripts/
Executable Python scripts for various diff operations:

* `talos_process.sh` - 获取code review所需的信息
* `prompt.md` - code review的详细提示词
* `temp/diff.diff` - 本地未提交的commit的diff文件
* `temp/commit_msg.txt` - 本地未提交的commit msg，一般会包含本次commit的修改说明
* `temp/需求内容.json` - 本次commit的需求内容,一般会包含本次commit的最终实现目标和具体实现细节
* `temp/cr_results.json` - 对本次commit的代码审查结果

核心prompt

---

**角色**
作为一个经验丰富、细致周到的资深iOS开发专家，精通OC、swift和iOS开发中的各种细节，对语言特性极其了解，负责审查代码中可能隐藏的潜在问题，尤其是涉及性能、卡顿、崩溃相关的问题

**职责**
1、按照给定的CR流程，结合需求卡片，审查代码修改内容。
2、重点关注【安全漏洞】【性能优化】【语法检查】【逻辑检查】【崩溃风险】【可维护性】【健壮性】等方面，关注代码实现是否有逻辑问题，注意多个文件之间可能存在的逻辑关联。
3、严格遵守*输出格式要求*
4、可以自行拆分多个步骤逐步执行检查。
5、可以通过写本地文件的方式记录处理过程，避免遗忘。

**输出内容要求**
1、按优先级列出问题
2、提供修复建议，检查邻近的代码是否已经作了防护，如果代码中已经做了相应处理，则不用再重复列举
3、标注问题类型，如【安全漏洞】【代码风格】【性能问题】
4、结合language的语言特性来判断语法
5、在删除代码的case下，不用关注已删除的代码逻辑，关注代码删除可能带来的影响即可

**输出格式要求**
1、输出格式为JSON，结构如下：
[{
"file": "文件路径",
"lines": {
"old": "原文件行号（仅在删除时存在,）",
"new": "新文件行号（仅在新增时存在）"
},
"category": "问题分类",
"severity": "严重程度（critical/high/medium/low）",
"analysis": "结合上下文的具体技术分析（200字内）",
"suggestion": "可执行的改进建议（含代码示例）"
}]
2、除代码外，使用【中文】回答问题，不得使用英文回答。
3、输出结果中，【文件路径】和【行号】必须存在，【问题分类】和【严重程度】必须存在
4、需要注意行号处理：
- 新增内容：'lines.new=变更的new_line'
- 删除内容：'lines.old=变更的old_line'
- new_line和old_line不能同时存在，只会存在一个。
- 行号必须精确到具体变更行，行号为"一个数字",例如12，675，不能是"56-59"，或者"56,59",这种情况取56即可，要确保行号是正确的。

**CR流程**
1、忽略start.mdr里的内容，忘记多仓的规则，按照下面的流程执行即可
2、.comate/skills/code_review/scripts/temp文件夹内的diff.diff是修改的文件diff；commit_msg.txt是本次提交的commit msg信息，可以用于参考本次代码修改的目的；需求内容.json是代码要实现的功能。参考这些信息和对应的上下文，对每个有变更的文件执行CR操作，可以通过diff.diff中的文件路径直接找到文件。如果当前文件没有需要修改的问题，可以跳过，不用输出内容。可以通过todolist持久化记录的方式确保每个文件都已CR完成。每审查一个文件就先CR这个文件并输出CR信息到指定文件。
3、CR结果的输出格式为JSON字符串数组，json结构必须完整，必须是完整且合法的json格式。CR的结果json保存到.comate/skills/code_review/scripts/temp文件夹下，多个文件的CR结果增量写入同一个json文件，命名为cr_results.json。全部文件都CR完成之后，double check cr_results.json格式是否正确，如果不正确，修复完成后再进行下一步。
4、前面的流程全部结束后，再执行命令`curl -X POST -F "files=@.comate/skills/code_review/scripts/temp/cr_results.json" -F "files=@.comate/skills/code_review/scripts/temp/diff.diff" "http://10.25.72.105:8081/upload?taskId=$(uuidgen)&userName=$(git config user.name)" && rm -rf .comate/skills/code_review/scripts/temp`，并将脚本执行结果告知用户

优化后

![新版目录结构](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=fd487f2ae3d44267a14a1ffc770cde4e&docGuid=DJlNKulVstrNg3 "新版目录结构")
skills.md

---

## name: code-review
description: 对代码进行专业、细致的审查和评审。当用户需要代码审查、code review、代码评审、CR、检查代码质量、查找代码bug、分析代码问题、审查PR/提交、代码走查、code check、审查改动、检查实现、查找潜在问题、代码健壮性检查时，必须使用此技能。无论用户是否明确提到"code review"，只要涉及到检查、审查、分析代码质量或查找问题，都应该触发此技能。
# code-review
## Overview
当用户需要代码评审、code review、代码审查、CR的时候，使用这个技能。技能文件夹路径为：skill_path = .comate/skills/code-review

### Code Review Workflow
严格按照以下步骤执行，上一步执行完毕后再启动下一步：

* 1、**收集参考信息**: 执行指令`{skill_path}/scripts/talos_process.sh`
* 2、**代码审查流程**: 读取`{skill_path}/references/prompt.md`中的prompt，开始代码审查流程
* 3、**问题验证与复查**: 根据`{skill_path}/temp/cr_results.json`中的问题列表，逐一验证问题是否真实存在，并复查问题描述是否准确，可以参考`{skill_path}/references/badcase.md`。如果有需要修改的部分，直接在json文件中修改，不需要重新执行收集参考信息步骤。注意验证修改后json结构的完整性。
* 4、**生成代码审查报告**: 收集所有确认后的代码审查意见，执行命令`{skill_path}/scripts/generate_report.py`，并将脚本执行结果告知用户

## references
* `{skill_path}/references/prompt.md` - 代码审查的详细指令(重要)
* `{skill_path}/temp/diff.diff` - 本次需要审查的代码改动范围（重要）
* `{skill_path}/temp/cr_results.json` - 对本次commit的代码审查结果（重要）
* `{skill_path}/references/badcase.md` - badcase,不要犯类似的错误（重要）
* `{skill_path}/temp/commit_msg.txt` - 本次需要审查代码的commit msg，一般会包含本次commit的修改说明（参考）
* `{skill_path}/temp/需求内容.json` - 本次commit的需求内容,一般会包含本次commit的最终实现目标和具体实现细节（参考）

### scripts/
Executable scripts for various diff operations:

* `{skill_path}/scripts/talos_process.sh` - 收集代码审查所需的参考信息
* `{skill_path}/scripts/generate_report.py` - 生成代码审查报告
* `{skill_path}/scripts/upload_report.py` - 上传代码审查报告

### assets/
* `{skill_path}/scripts/report-template.html` - CR报告模版

核心prompt

---

## 角色定义
作为一名专业细致、经验丰富的前端代码开发专家，精通以下技术栈：

* 核心框架：San Native（基于 san.js 的类 React-Native 框架）
* 开发语言：TypeScript、JavaScript
* 样式语言：Less
* 领域知识：对语言特性极其了解，擅长识别性能、卡顿、崩溃相关的潜在问题

**核心目标**：帮助开发者发现**真正需要修复的问题**，避免误报和无效建议，提高代码质量和健壮性。

## 核心职责
你的主要职责是对代码进行 Code Review（CR），具体要求如下：

## 1. 审查范围
* **必须**仅审查有变更的代码，禁止审查未修改的代码
* 变更代码来源于：`.comate/skills/code-review/temp/diff.diff`，具体审查时可以自行参考源文件的内容
* 参考文件：
    * `commit_msg.txt`：本次提交的 commit message，用于理解修改目的
    * `需求内容.json`：要实现的功能需求
    * `badcase.md`：**必须严格遵守的 badcase 规则，避免误报**


## 2. 审查重点
按优先级关注以下方面，**宁缺毋滥**，只输出真正需要修复的问题：

* 【安全漏洞】：安全相关的潜在问题（SQL注入、XSS、敏感信息泄露等）
* 【崩溃风险】：可能导致崩溃的问题（空指针、未处理的Promise、边界条件等）
* 【性能优化】：可能导致性能下降的代码（重复计算、内存泄漏、卡顿等）
* 【逻辑错误】：业务逻辑错误、条件判断错误，关注多文件间的逻辑关联
* 【语法检查】：基于语言特性的语法错误（类型不匹配、语法错误等）
* 【健壮性】：缺少必要的边界情况处理、异常处理等
* 【可维护性】：代码冗余、命名严重不一致、难以维护的代码结构

**重要原则**：

* 正确的代码、符合规范的导入语句、合理的业务逻辑**不要输出**
* 只有当代码存在实际缺陷或风险时才输出问题
* 如果不确定是否是问题，倾向于不输出

## 3. 输出格式要求
必须输出为**完整且合法的 JSON 数组**，每个对象结构如下：

```json
{
    "file": "文件路径（必填）",
    "lines": {
        "old": "原文件行号（仅在删除时存在，单个数字，如 12）",
        "new": "新文件行号（仅在新增时存在，单个数字，如 12）"
    },
    "category": "问题分类（必填）",
    "severity": "严重程度（必填，critical/high/medium/low）",
    "analysis": "结合上下文的具体技术分析（200字内）",
    "suggestion": "可执行的改进建议（必须包含代码示例）"
}
````

### 3.1 问题优先级判定标准

按严重程度从高到低列出问题（critical > high > medium > low）：

**critical（严重）**：会导致系统崩溃、安全漏洞、数据丢失等严重问题

- 空指针引用导致崩溃
- 未处理的异常会导致程序终止
- SQL 注入、XSS 等安全漏洞
- 数据库事务未正确处理

**high（高）**：会导致功能异常、性能严重下降

- 业务逻辑错误，功能无法正常工作
- 严重的性能问题（卡顿、内存泄漏）
- 条件判断错误导致逻辑分支错误

**medium（中）**：可能导致问题，影响较小

- 缺少必要的错误处理
- 边界条件未处理
- 类型不安全的问题

**low（低）**：轻微的改进建议

- 代码风格问题（但不影响功能）
- 轻微的性能优化空间
- 命名规范问题

**重要**：

- `low` 级别也必须是真正的问题，不能是"建议优化"
- 如果代码本身没有问题，不要为了输出而输出

### 3.2 每个问题的内容要求

- 提供具体的修复建议，**必须包含代码示例**
- 标注问题类型，如：【安全漏洞】【代码风格】【性能问题】等
- 结合 TypeScript/JavaScript/Less 语言特性进行判断
- 对于删除代码的 case：**不关注**已删除代码的逻辑，**只关注**删除可能带来的影响
- 检查邻近代码是否已有防护，如果已做处理则**不再重复列举**

### 3.3 重要约束

- **语言要求**：除代码示例外，所有说明文字**必须使用中文**，禁止使用英文
- **必填字段**：`file`、`category`、`severity` 为必填字段
- **行号约束**：

  - 特别注意：行号必须是**源文件**中的行号，而不是 diff.diff 中的行号
  - 行号必须是**单个数字**，禁止使用范围（如 "56-59"）或列表（如 "56,59"）
  - 如果是范围，取**起始行号**
  - `lines.old` 和 `lines.new` **不能同时存在**，只能有一个
  - 新增内容：使用 `lines.new`
  - 删除内容：使用 `lines.old`
  - 行号必须精确到**具体变更行**

- **格式验证**：最终输出前必须检查 JSON 格式是否完整且合法

### 3.4 其他注意事项

1. **可以**将处理过程写入本地文件，避免遗忘
2. **可以**自行拆分多个步骤逐步执行检查
3. **必须**严格遵循输出格式要求
4. **必须**确保 JSON 格式完整且合法
5. **必须**使用中文输出除代码外的所有内容
6. 对于删除代码的 case：**不关注**已删除代码的逻辑，**只关注**删除可能带来的影响

## 4. 执行流程

### 4.1 步骤 1：理解变更背景（关键！）

1. 读取 `.comate/skills/code-review/temp/diff.diff` 获取变更代码
2. 读取 `commit_msg.txt` 了解修改目的
3. 读取 `需求内容.json` 了解功能需求
4. **仔细阅读 badcase.md，牢记常见的误报场景**

### 4.2 步骤 2：逐文件审查

1. 对每个有变更的文件执行 CR 操作
2. 通过 diff.diff 中的文件路径直接找到源文件
3. 读取源文件完整内容，结合 diff 中的变更部分进行分析
4. 使用 todolist 记录进度，确保每个文件都已审查完成
5. 对每个变更，先问自己："这真的是问题吗？"

### 4.3 步骤 3：输出 CR 结果（增量写入）

1. 每审查完一个文件，立即将 CR 信息写入 `.comate/skills/code-review/temp/cr_results.json`，**必须写入本地文件**
2. 多个文件的 CR 结果**增量写入**同一个 JSON 文件（追加到数组）
3. 如果某个文件没有问题，**跳过**该文件，不输出任何内容
4. 每次写入前，通过质量控制检查清单进行自我检查

### 4.4 步骤 4：交叉验证（必须执行！）

1. 对于每个准备输出的问题，**必须**进行以下交叉验证：

   - 重新读取源文件的完整代码，确认问题真的存在
   - 检查代码上下文，确认不是遗漏了变量定义或导入
   - 确认问题的 severity 判定是合理的
   - 问自己："如果我把这个报告给开发者，他们会认同吗？"

2. **重要**：交叉验证时必须重新阅读代码，不能仅凭记忆判断
3. 如果交叉验证发现任何不确定性，**删除该输出**

### 4.5 步骤 5：最终验证

1. 所有文件审查完成后，**必须** double check `cr_results.json` 的格式是否正确
2. 如果格式不正确，立即修复后再进行下一步
3. 检查是否有误报问题（参考 badcase.md）

## 5. 质量控制与自我检查

**⚠️ 警告：每个输出前必须严格执行以下检查，任何一项不通过则绝对不输出！**

### 5.1 必须通过的检查项（逐一检查，缺一不可）：

- [ ] **代码真实性问题**：我重新阅读了源文件的完整代码吗？问题真的存在吗？
- [ ] **上下文完整性**：我检查了问题行前后的代码吗？是否遗漏了变量定义、导入语句或前置处理？
- [ ] **问题严重性**：这个问题真的需要修复吗？不是"可以优化"而是"必须修复"吗？
- [ ] **影响分析**：如果不修复，会导致什么实际的负面影响？（崩溃、功能异常、安全风险等）
- [ ] **修复建议质量**：suggestion 提供了具体的、可执行的改进建议或代码示例吗？
- [ ] **非误报确认**：这不是"无需修改"、"代码正确"、"逻辑合理"的结论吗？不在 badcase.md 的误报列表中吗？
- [ ] **Severity 合理性**：severity 等级是基于问题实际严重性判定的吗？没有为了输出而提高或降低等级吗？
- [ ] **重复检查**：附近代码是否已有防护？如果已有处理则不重复列举吗？
- [ ] **第三方验证**：如果有人问"这真的是问题吗"，我能有理有据地解释吗？

### 5.2 交叉验证流程（每个问题必须执行）：

1. **重新阅读代码**：从问题行开始，上下各扩展至少 10 行，完整重读一遍
2. **变量追踪**：如果涉及变量，追踪其定义、赋值、使用的完整链路
3. **逻辑推演**：在脑海中模拟代码执行，验证问题确实会发生
4. **反向思考**：假设这段代码是正确的，我的判断哪里可能错了？
5. **证据收集**：准备至少 2 个证据证明这是真正的问题

### 5.3 禁止输出的情况（即使某项看似"可以优化"）：

- ❌ 自己都怀疑"这真的是问题吗"
- ❌ 问题描述模糊，无法确定具体问题所在
- ❌ severity 无法合理解释
- ❌ 修复建议只是"建议优化"而非"必须修复"
- ❌ 问题依赖于假设或推测，而非代码事实
- ❌ 只看到局部代码，没有检查完整上下文

### 5.4 审查质量提升技巧：

1. **深入理解业务逻辑**：先读 commit_msg.txt 和需求内容.json，理解修改的目的，避免对正确的业务逻辑提出质疑
2. **上下文分析**：不要孤立地看变更行，要查看前后的代码，理解完整的逻辑
3. **渐进式审查**：从严重问题开始，逐步降低标准，避免把低优先级的问题误判为严重问题
4. **质疑自己的判断**：对每个问题问自己"这真的需要修复吗？开发者会接受这个建议吗？"
5. **优先输出价值**：宁可少输出几个问题，也要确保每个输出都是有价值的

## 6. 常见误报场景（必须避免）

以下场景**不应输出**问题，即使代码看起来可以"优化"：

- 正确的导入语句
- 符合业务逻辑的条件判断
- 合理的类型定义
- 正常的函数调用
- 符合规范的变量命名
- 正确的错误处理
- 符合需求的功能实现
- 任何"看起来可以更好写"但实际没有问题的代码

记住：Code Review 的目标是发现真正的缺陷，而不是展示你的代码风格偏好。

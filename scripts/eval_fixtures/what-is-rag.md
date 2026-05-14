RAG 实战分享：从向量检索到 LightRAG 知识图谱

> **面向开发者的 RAG 技术解析**
> 分享时长：45 ~ 60 min | 难度：入门 → 进阶

---

## 目录

1. [破冰：你每天都在用 RAG，但你不知道](#1)
2. [为什么 LLM 本身不够用](#2)
3. [RAG 是什么：核心思想与整体架构](#3)
4. [离线流程 Part 1：文档预处理](#4)
5. [离线流程 Part 2：切块（Chunking）](#5)
6. [离线流程 Part 3：Embedding 向量化](#6)
7. [在线流程 Part 1：Query 理解与增强](#7)
8. [在线流程 Part 2：向量检索与混合检索](#8)
9. [在线流程 Part 3：Rerank 精排](#9)
10. [在线流程 Part 4：Prompt 构建与生成](#10)
11. [RAG 效果评估体系](#11)
12. [LightRAG：知识图谱驱动的下一代 RAG](#12)

---

## 1. 破冰：你每天都在用 RAG，但你不知道

### 1.1 几个你熟悉的场景

作为一名程序员，你可能用过或见过这些工具：

**场景 A：Comate / Ducc 读懂你的项目**

你在一个几百万行的 Android 项目里问 Comate：「UBCManager 如何完成打点？」它能给你一个结合项目实际代码的回答，而不是泛泛而谈。

这不是魔法——它在你问之前，已经把你的代码库做了索引，问的时候先检索相关代码，再喂给 LLM。

**场景 B：百度内部智能客服 / 知识库问答**

公司把所有产品文档、FAQ、工单记录放进知识库，配合如流智能助手，员工问「XX 功能怎么用」或「XX 系统报错怎么处理」，它能精准检索知识库文档并给出答案，不会乱说。

背后同样是 RAG——先检索相关文档，再喂给 LLM 生成回答。

**场景 C：你们团队可能正在做的事**

如果你是一名研发人员，团队里一定有 iCafe 需求卡片、iCode 代码仓库、知识库文档、iAPI 接口文档等。

老板或许已经在问：「能不能让 AI 直接读这些，帮我们快速查需求、看代码、找接口？」

——这正是 RAG 在研发场景的典型落地，把分散在各平台的研发数据统一索引，让 AI 成为你的「研发助手」。

**场景 D：你用过的 Skill / Plugin 体系**

在某些 AI 工具平台上，你写了一个「自动上车」的 Skill，本质上也是在做有针对性的信息检索和上下文注入，和 RAG 的思路一脉相承。

### 1.2 今天我们要搞懂什么

```
LLM 为什么不够用？
        ↓
RAG 的整体架构是什么？
        ↓
数据怎么进去：预处理 → 切块 → Embedding → 向量存储
        ↓
问题怎么出来：Query增强 → 检索 → Rerank → 生成
        ↓
为什么还需要知识图谱？LightRAG 解决了什么问题？
```

---

**RAG = Retrieval-Augmented Generation（检索增强生成）**

> **一句话理解**：
> **在大模型回答问题之前，先从外部知识库中“查资料”，再基于查到的资料生成答案。**

## 2. 为什么 LLM 本身不够用

#### 2.1 LLM 的三大硬伤

先建立一个直觉：**LLM 是一个「博闻强记但有截止日期且不了解你公司」的顾问**。

| **问题**                  | **具体表现**                                             | **移动端开发类比**                                                            |
| ------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **知识截止日期**          | 训练数据到某个时间点为止，不知道之后发布的 SDK、API 变更 | 一本出版于两年前的《Android 开发实战》，里面还在讲旧版 API                    |
| **幻觉（Hallucination）** | 自信地生成不存在的函数名、参数、返回值                   | 一个同事说「直接调 `setTranslucentStatusBar()` 就行」，结果这个方法根本不存在 |
| **私域数据盲区**          | 对你公司的代码库、设计文档、内部 SDK 一无所知            | 新来的外包，连你们的 Git 仓库都没权限                                         |

#### 2.2 幻觉有多严重？一个真实案例

斯坦福大学一项研究（2023）对主流 LLM 在医疗/法律场景的准确率做了测试

在需要精确引用具体文档时，幻觉率高达 **27%~46%**。

对程序员来说，幻觉意味着：

        * 给你一个看起来能跑但实际崩溃的 API 调用
        * 引用一个不存在的 Android API Level
        * 说某个第三方库「支持 Kotlin Multiplatform」，其实根本不支持

#### 2.3 Context Window 不是银弹

**「我直接把所有文档塞进 Prompt 不就行了？」**

这是很多人第一直觉，也是需要当场打破的误区。

**Token 成本问题：**

> 📌 假设条件：10 万字文档 ≈ 150,000 tokens，输出量与输入量同级估算（[https://console.bce.baidu.com/support/?u=bce-head#/tokenizer](https://console.bce.baidu.com/support/?u=bce-head#/tokenizer)）
> |**维度**|**GLM-5**|**Kimi-K2.5**|**MiniMax-M2.5**|
> |-|-|-|-|
> |输入价格（元/千 tokens）|0.006|0.004|0.0021|
> |输出价格（元/千 tokens）|0.022|0.021|0.0084|
> |单次文档成本（元/次）|4.2|3.75|**1.58**|
> |每日成本（100 次/天）|420 元|375 元|**157.5 元**|
> |每月成本（30 天）|12,600 元|11,250 元|**4,725 元**|

---

> 💡 **MiniMax-M2.5 成本最低**，相比 GLM-5 每月可节省约 **7,875 元**，相比 Kimi-K2.5 每月可节省约 **6,525 元**
> **更本质的问题——Lost in the Middle：**

2023 年 Stanford 的论文《Lost in the Middle: How Language Models Use Long Contexts》做了一个实验：

把正确答案放在不同位置（开头、中间、结尾）的长 Context 里，测试 LLM 的回答准确率：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=0207eab21ff348059ac53aad89117f44&docGuid=umHNMyON7lLOVt)
**结论：你把 1000 页文档塞给模型，它很可能漏掉中间的关键信息。**

这不是 LLM 的 bug，这是 Transformer 注意力机制的固有特性

——在极长序列中，中间位置的 Token 获得的注意力权重更低。

#### 2.4 Fine-tuning 能解决吗？

常见误区：「那我 Fine-tune 一个专属模型不就好了？」

| **对比维度** | **Fine-tuning**                       | **RAG**                              |
| ------------ | ------------------------------------- | ------------------------------------ |
| **知识更新** | 需要重新训练，周期长（天到周）        | 向量库增量更新，实时生效（分钟级）   |
| **成本**     | 训练成本高昂（GPU/云服务）            | 主要成本是 Embedding（便宜）         |
| **可解释性** | 模型内部，无法溯源                    | 可以标注「答案来自第 X 文档第 Y 段」 |
| **适合场景** | 改变模型「风格/能力」（如更会写代码） | 注入「特定知识」（如公司内部文档）   |
| **幻觉控制** | 仍然会幻觉                            | 可以限制「只基于检索结果回答」       |

**结论：Fine-tuning 和 RAG 解决的是不同问题，不是替代关系，高质量系统往往两者结合。**

---

## 3. RAG 是什么：核心思想与整体架构

#### 3.1 一句话定义

**RAG = 在调用 LLM 生成答案之前，先从外部知识库中检索出最相关的信息，把它们拼进 Prompt，让 LLM 基于这些信息来回答。**

#### 3.2 类比：考试开卷 vs 闭卷

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=eb1200f88b424d0a93d5d08155b72136&docGuid=umHNMyON7lLOVt)
**关键点：开卷考试的质量**

**70% 取决于你找资料的能力，30% 取决于你用资料答题的能力**。

RAG 里检索质量的重要性远超大多数人的预期。

#### 3.3 RAG 全链路架构

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=b99e2ce401d9455a9df81b979c7dd400&docGuid=umHNMyON7lLOVt)
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=0668a583e8064d6399ca32b950ae966d&docGuid=umHNMyON7lLOVt)
RAG 的核心思想很简单：让模型‘先查资料，再回答问题’。它的架构分为 **离线流程**（准备知识库）和 **在线流程**（实时查询响应）

也有部分文档将其分为 **索引阶段 Indexing** 和 **在线问答阶段 Retrieve + Generate**

接下来我们逐个环节深入。

---

## 4. 离线流程 Part 1：文档预处理

对于原始文档，根据内容格式可分为**结构化文档**和**非结构化文档**，两者在预处理流程上存在显著差异。

#### 4.1 为什么预处理至关重要

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=b65ac88633234c8a9681d15200954bc7&docGuid=umHNMyON7lLOVt)
**「垃圾进，垃圾出（Garbage In, Garbage Out）」**

——这是机器学习领域最朴素的真理，在 RAG 里体现得格外明显。

**不做预处理的后果：**

```
原始 PDF 解析结果（未处理）：
"第 3 章 \n \n 用户认证模 \n块设计 \n\n 3.1 概述 \n ...(页眉)公司内部文档...(页码)23"

Embedding 之后：向量里夹杂着页眉、页码、断行的「认证模\n块」
检索时：模糊匹配到「认证模块」的概率大幅下降
```

#### 4.2 不同文档类型的处理策略

##### 4.2.1 PDF 文档

PDF 是移动端团队最常见的文档格式，也是最难处理的格式之一。

**工具选择：**

            * 方案 1：pdfplumber（推荐，保留结构）
            * 方案 2：PyMuPDF（速度快，适合大量文档）
            * 方案 3:  借助LLM（图像转文本），[使用 Markdown 和 Gemini 为 RAG 解锁 PDF](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/2KxRNBslIc/6EqR4e8xE4E2ZX?t=mention&mt=doc&dt=doc)

##### 4.2.2 Markdown / 知识库文档

技术团队的文档通常是 Markdown 或 知识库，这类文档结构清晰，是最友好的格式。

##### 4.2.3 代码文件

代码是移动端团队的核心资产，处理策略和普通文档完全不同。

        * 代码文件：按函数/类边界切，而不是按 token 数切

```python
# 代码文件：按函数/类边界切，而不是按 token 数切
import ast
import javalang  # Java/Android

def extract_kotlin_functions(kotlin_code: str) -> list[dict]:
    """提取 Kotlin 函数，每个函数作为独立单元"""
    functions = []

    # 简化示例：按 fun 关键字分割
    pattern = r'((?:/\*\*.*?\*/\s*)?(?:@\w+\s*)*(?:fun\s+\w+[^{]*\{[^}]*(?:\{[^}]*\}[^}]*)*\}))'
    matches = re.findall(pattern, kotlin_code, re.DOTALL)

    for match in matches:
        functions.append({
            "content": match,
            "type": "function",
            "language": "kotlin"
        })
    return functions
```

        * 更好的方案：tree-sitter（支持多语言 AST 解析）

```python
pip install tree-sitter tree-sitter-kotlin tree-sitter-swift
```

**代码 + 注释联合处理的最佳实践：**

```kotlin
// 原始代码
/**
 * 获取用户信息，包含重试机制
 * @param userId 用户唯一标识
 * @return 用户信息，如果获取失败返回 null
 */
suspend fun getUserInfo(userId: String): UserInfo? {
    return retryIO(times = 3) {
        apiService.getUser(userId)
    }
}
```

对这段代码，最好的索引方式是：**将函数签名 + KDoc/JavaDoc 注释作为主要语义载体**，代码体作为补充。这样 Embedding 的语义质量更高。

##### 4.2.4 HTML / 网页文档

````python
from readability import Document  # pip install readability-lxml
import requests
from bs4 import BeautifulSoup

def extract_main_content(url: str) -> str:
    response = requests.get(url)

    # 使用 readability 提取主要内容（去除导航、广告、侧边栏）
    doc = Document(response.text)
    main_html = doc.summary()

    # 转纯文本
    soup = BeautifulSoup(main_html, 'html.parser')

    # 保留代码块
    for code in soup.find_all(['code', 'pre']):
        code.string = f"\n```\n{code.get_text()}\n```\n"

    return soup.get_text(separator='\n', strip=True)
````

#### 4.3 Metadata 设计——经常被忽视的关键

**每个文档块不只要存内容，还要存 Metadata。**

Metadata 的作用：

在检索时做**过滤（Filter）**，在生成时做**溯源（Citation）**。

```python
# 好的 Metadata 设计
metadata = {
    # 来源信息
    "source_file": "android_network_guide.md",
    "source_url": "https://wiki.internal/android/network",

    # 文档层级
    "doc_type": "technical_guide",       # 文档类型
    "module": "network",                 # 模块
    "chapter": "错误处理",               # 章节

    # 版本/时间
    "version": "Android 14",
    "updated_at": "2024-03-15",
    "author": "张工",

    # 内容特征
    "language": "zh",
    "content_type": "text",             # text / code / table
    "has_code_example": True,

    # 切块位置
    "chunk_index": 3,
    "total_chunks": 12,
}
```

**Metadata 驱动的精准检索：**

```python
# 用户问关于 Android 14 网络模块的问题
results = vectorstore.similarity_search(
    query="如何处理网络超时重试",
    filter={
        "module": "network",
        "version": "Android 14",
        "content_type": "text"
    },
    k=10
)
# 大幅减少噪声，精准度提升
```

---

## 5. 离线流程 Part 2：切块（Chunking）

#### 5.1 切块的本质问题

切块要解决一个根本矛盾：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=c642a70bfea04fdaadea93c49dd498e8&docGuid=umHNMyON7lLOVt)
**没有万能的 chunk_size，只有适合你数据的 chunk_size！**

#### 5.2 五种主流切块策略深度对比

##### 策略 1：固定长度切块（Fixed Size Chunking）

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,      # 每块最多 500 个字符
    chunk_overlap=50,    # 相邻块重叠 50 个字符
    separator="\n"
)
chunks = splitter.split_text(text)
```

**缺点演示：**

```
原文：
    "...用户点击登录按钮后，系统会验证用户名和密码。如果验证通过，
    将生成一个 JWT Token 并返回给客户端..."

切块结果（假设恰好在「JWT」处切断）：
    块A: "...用户点击登录按钮后，系统会验证用户名和密码。如果验证通过，将生成一个 J"
    块B: "WT Token 并返回给客户端..."

问题：搜索「JWT Token 生成逻辑」时，两块都匹配不上，因为关键词被切断了
```

`chunk_overlap` 的意义就是解决这个问题：

```
有了 overlap：
    块A: "...将生成一个 JWT Token 并返回" ← overlap 区域包含了完整的关键信息
    块B: "JWT Token 并返回给客户端..."    ← 同样包含完整语义
```

##### 策略 2：递归字符切块（Recursive Character Splitter）—— 最推荐

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 按优先级尝试不同分隔符：先按段落，再按句子，再按词，最后硬切
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=[
        "\n\n",   # 优先在段落边界切
        "\n",     # 其次在行边界切
        "。",     # 中文句号
        ".",      # 英文句号
        "！", "？", "!", "?",
        " ",      # 空格
        ""        # 最后硬切
    ]
)
```

这个策略的精妙之处：**尽量保持语义完整，只在迫不得已时才硬切。**

##### 策略 3：语义切块（Semantic Chunking）

不按固定长度切，而是检测语义边界——相邻句子的 Embedding 相似度突然下降，说明话题转换了，在这里切。

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",   # 用百分位数判断语义断点
    breakpoint_threshold_amount=95            # 相似度低于第 95 百分位时切块
)
chunks = splitter.split_text(long_document)
```

**优缺点：**

    * ✅ 块的语义最完整，每块聚焦一个话题
    * ❌ 需要对整个文档做 Embedding（额外成本），块大小差异大

##### 策略 4：按文档结构切块（Structure-aware Chunking）

```python
def chunk_by_markdown_headers(md_text: str) -> list[dict]:
    """按 Markdown 标题层级切块"""
    from langchain.text_splitter import MarkdownHeaderTextSplitter

    headers_to_split_on = [
        ("#",  "h1"),
        ("##", "h2"),
        ("###","h3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False    # 保留标题，作为块内容的一部分
    )

    splits = splitter.split_text(md_text)
    # 每个 split 自动带有 {h1, h2, h3} 的 metadata
    return splits
```

**适合场景：** API 文档、设计规范、有清晰章节结构的技术文档。

##### 策略 5：Parent-Child 双层切块（进阶）

这是一个很优雅的设计，解决了「检索粒度」和「上下文完整性」的矛盾：

```
Parent Chunk（大块，512~1024 tokens）—— 存储用，提供完整上下文
        ↓ 包含
Child Chunk（小块，128~256 tokens）—— 检索用，精准匹配

查询流程：
  1. 用 Child Chunk 做精细检索（粒度小，命中精准）
  2. 找到 Child Chunk 后，返回对应的 Parent Chunk（上下文完整）
  3. 把 Parent Chunk 喂给 LLM（信息量更完整）
```

### 5.3 不同场景的切块参数建议

| **文档类型** | **推荐策略**    | **chunk_size**  | **overlap** | **原因**                        |
| ------------ | --------------- | --------------- | ----------- | ------------------------------- |
| 中文技术文档 | Recursive       | 400~600 字      | 60~100 字   | 中文信息密度高，不需要太大      |
| 英文文档     | Recursive       | 512~1024 tokens | 64~128      | 英文需要更多 token 表达相同信息 |
| API 参考文档 | Structure-aware | 按接口切        | 无          | 每个接口是独立语义单元          |
| 代码文件     | 按函数/类切     | 按结构          | 无          | 代码结构即语义边界              |
| FAQ 问答     | 按问答对切      | 一问一答        | 无          | 问答对是完整语义单元            |
| 需求文档     | Parent-Child    | P:800 C:200     | P:80 C:20   | 需要精准匹配 + 完整上下文       |

Matrix LightRAG 采用 **"两级语义切分 + 滑动窗口字符切分 + 结构保护"** 的复合切分策略。先通过 H1/H2 标题保证语义完整性，再在语义块内使用 1500 字的滑动窗口（重叠 50 字）进行细粒度切分，同时通过保护机制确保代码块、表格、链接等结构不被破坏。

#### 5.4 如何验证切块质量

不要靠感觉，要有客观验证方法：

```python
def evaluate_chunks(chunks: list[str], test_queries: list[str]):
    """
    对一批测试问题，检验切块后能否找到答案
    """
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('BAAI/bge-m3')
    chunk_embeddings = model.encode(chunks)

    for query in test_queries:
        query_embedding = model.encode(query)
        scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

        top_k = scores.topk(3)
        print(f"\nQuery: {query}")
        for score, idx in zip(top_k.values, top_k.indices):
            print(f"  Score: {score:.3f} | Chunk: {chunks[idx][:100]}...")

        # 人工判断：Top-3 里是否包含正确答案？
        # 如果经常找不到 → 调小 chunk_size 或换策略
        # 如果找到了但内容不完整 → 调大 chunk_size 或增大 overlap
```

---

## 6. 离线流程 Part 3：Embedding 向量化

### 6.1 Embedding 的数学直觉

**一个向量 = 一个语义坐标**

想象一个超高维空间（768 维或 1536 维），每个文本片段都是这个空间里的一个点。**语义相近的文本，在这个空间里距离更近。**

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=5329fca6d7c64d598a680c20d4cb8057&docGuid=umHNMyON7lLOVt)
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=e1871ddbcffe42bf9720140d3c983722&docGuid=umHNMyON7lLOVt)
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=3e03c86472ef456bb31791cf83053099&docGuid=umHNMyON7lLOVt)
**余弦相似度（Cosine Similarity）：** 衡量两个向量方向的相似程度，值域 [-1, 1]，越接近 1 表示越相似。

```python
import numpy as np

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

# 示例
sim_1 = cosine_similarity(embed("ANR 问题"), embed("App 无响应"))
sim_2 = cosine_similarity(embed("ANR 问题"), embed("今天天气"))
print(sim_1)  # 约 0.89 —— 高度相似
print(sim_2)  # 约 0.15 —— 几乎无关
```

### 6.2 Embedding 模型的工作原理（简版）

现代 Embedding 模型（如 BGE、text-embedding-3）基于 **双塔 Transformer 架构**：

```
输入文本
    ↓
Tokenizer（分词，转换为 Token ID 序列）
    ↓
Transformer Encoder（多层自注意力，捕捉 Token 间关系）
    ↓
Pooling（把所有 Token 的向量聚合为一个向量）
    ├── CLS Token Pooling（用 [CLS] Token 的向量）
    ├── Mean Pooling（所有 Token 向量取平均）
    └── Max Pooling（每维取最大值）
    ↓
可选：L2 归一化（让向量长度为 1，方便余弦相似度计算）
    ↓
输出：固定维度的浮点向量 [v1, v2, ..., v768]
```

```python
┌─────────────────────────────────────────────────────────────────┐
│                         输入文本                                 │
│                      "机器学习很有趣"                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ① Tokenizer 分词                                               │
│  ─────────────────────────────────────────────────────────────  │
│  作用：把文字变成数字（计算机只认数字）                            │
│                                                                 │
│  "机器学习很有趣"                                                │
│       ↓                                                         │
│  ["机器", "学习", "很", "有趣"]                                   │
│       ↓                                                         │
│  [2001, 3002, 500, 4003]  ← 每个词的"身份证号"                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ② Transformer Encoder 编码                                     │
│  ─────────────────────────────────────────────────────────────  │
│  作用：让每个词"看到"其他词，理解上下文语义                         │
│                                                                 │
│  🔑 核心：自注意力机制                                           │
│     每个词都在问："我该关注句子里的哪些词？"                        │
│                                                                 │
│  示例：同样的"苹果"                                              │
│  ┌─────────────────┬─────────────────┐                         │
│  │  "苹果很好吃"    │ "苹果发布新手机" │                         │
│  │       ↓         │        ↓        │                         │
│  │  向量偏向"水果"  │  向量偏向"公司"  │                         │
│  └─────────────────┴─────────────────┘                         │
│                                                                 │
│  输出：每个词一个向量                                             │
│  "机器" → [0.2, 0.8, 0.1]                                       │
│  "学习" → [0.3, 0.7, 0.2]                                       │
│  "很"   → [0.1, 0.1, 0.5]                                       │
│  "有趣" → [0.4, 0.3, 0.6]                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ③ Pooling 池化聚合                                             │
│  ─────────────────────────────────────────────────────────────  │
│  作用：N 个词向量 → 合并成 1 个句子向量                           │
│                                                                 │
│  ┌───────────────┬────────────────────┬───────────────────┐    │
│  │    方法        │       做法         │       比喻        │    │
│  ├───────────────┼────────────────────┼───────────────────┤    │
│  │ CLS Pooling   │ 用 [CLS] 标记向量   │ 班长代表全班发言   │    │
│  │ Mean Pooling  │ 所有向量取平均      │ 全班成绩取平均分   │    │
│  │ Max Pooling   │ 每维度取最大值      │ 每科取最高分组队   │    │
│  └───────────────┴────────────────────┴───────────────────┘    │
│                                                                 │
│  Mean Pooling 示例：                                            │
│  ([0.2,0.8,0.1] + [0.3,0.7,0.2] + [0.1,0.1,0.5] + [0.4,0.3,0.6])│
│  ÷ 4 = [0.25, 0.475, 0.35]                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ④ L2 归一化（可选）                                             │
│  ─────────────────────────────────────────────────────────────  │
│  作用：把向量长度压缩为 1，方便计算余弦相似度                       │
│                                                                 │
│  原向量:  [3, 4]     长度 = √(3²+4²) = 5                        │
│  归一化:  [0.6, 0.8] 长度 = 1                                    │
│                                                                 │
│  ✅ 好处：点积 = 余弦值，计算更快                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       📤 输出向量                                │
│              [0.40, 0.76, 0.56, ..., 0.23]                      │
│                    （通常 768 或 1536 维）                        │
│                                                                 │
│              这个向量代表整句话的"语义位置"                        │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 主流 Embedding 模型详细对比

| **模型名称**                  | **维度**             | **最大输入长度** | **价格（近似/示例）**                  | **特点**                                         |
| ----------------------------- | -------------------- | ---------------- | -------------------------------------- | ------------------------------------------------ |
| OpenAI text-embedding-3-large | 3072（可调低至 256） | 8191 tokens      | $0.13 / 1M tokens（输入）              | 高精度、支持维度调整、多语言、API 调用           |
| jina-embeddings-v2            | 768                  | 8192 tokens      | 开源免费                               | 多语言、长文本优化、开源可部署                   |
| multilingual-e5-large         | 1024                 | 514 tokens       | 开源免费                               | 多语言、基于 E5 架构、平衡性能与效率             |
| **Qwen/Qwen3-Embedding-8B**   | **8192**             | **8192 tokens**  | **开源免费**                           | **最新一代、性能大幅提升、长文本支持、中文优化** |
| **Qwen/Qwen3-Embedding-4B**   | **4096**             | **8192 tokens**  | **开源免费**                           | **中等规模、高性能、中文优化**                   |
| **Qwen/Qwen3-Embedding-2B**   | **2048**             | **8192 tokens**  | **开源免费**                           | **轻量级但性能强劲、中文优化**                   |
| Qwen/Qwen2-Embedding-8B       | 8192                 | 8192 tokens      | 开源免费                               | 高维度、大模型、长文本支持、中文优化             |
| Qwen/Qwen2-Embedding-4B       | 4096                 | 8192 tokens      | 开源免费                               | 中等规模、高性价比、中文优化                     |
| Qwen/Qwen2-Embedding-0.6B     | 768                  | 8192 tokens      | 开源免费                               | 轻量级、快速推理、中文优化                       |
| BAAI/bge-m3                   | 1024                 | 8192 tokens      | 开源免费                               | 多语言、多粒度、多任务、支持密集与稀疏检索       |
| BAAI/bge-large-zh-v1.5        | 1024                 | 512 tokens       | 开源免费                               | 中文优化、高精度、广泛用于中文任务               |
| 智谱 AI Embedding-3           | 1024                 | 8192 tokens      | 需联系获取商业价格（通常按调用量计费） | 中文优化、长文本支持、商业 API 服务              |

### 6.4 向量数据库选型

| **数据库**   | **类型**        | **适用场景**  | **核心优势**                    | **注意事项**                  | \*\*\*\* |
| ------------ | --------------- | ------------- | ------------------------------- | ----------------------------- | -------- |
| **Chroma**   | 本地嵌入式      | 开发调试、PoC | 零配置，Pythonic API            | 不适合生产，无分布式          |          |
| **FAISS**    | 本地库          | 研究、原型    | 速度极快，Facebook 出品         | 纯内存，无持久化，无 HTTP API |          |
| **Qdrant**   | 独立服务        | 生产推荐      | 高性能，支持复杂过滤，Rust 实现 | 需要独立部署                  | 大白     |
| **Weaviate** | 独立服务        | 混合检索      | 原生支持向量+BM25 混合检索      | 配置稍复杂                    |          |
| **Milvus**   | 分布式          | 超大规模      | 10 亿级向量，高可用             | 运维成本高                    | 需求助手 |
| **pgvector** | PostgreSQL 插件 | 已有 PG 团队  | 复用现有基础设施                | 性能不如专用向量库            |          |
| **Pinecone** | 托管云服务      | 快速上线      | 全托管，无运维                  | 成本较高，数据出境            |          |

### 6.5 Embedding 的几个重要细节

**细节 1：Query 和 Document 必须用同一个模型**

```python
# ❌ 错误：不同模型的向量空间完全不兼容
doc_vectors = bge_embed(documents)    # bge-m3 的 1024 维空间
query_vector = openai_embed(query)    # OpenAI 的 1536 维空间
# 余弦相似度计算结果毫无意义

# ✅ 正确
doc_vectors = bge_embed(documents)
query_vector = bge_embed(query)       # 同一模型，同一空间
```

**细节 2：批量 Embedding 要分批处理**

```python
def batch_embed(texts: list[str], batch_size=32) -> list[list[float]]:
    """分批处理，避免内存溢出和 API 限速"""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        try:
            embeddings = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True  # L2 归一化
            )
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"Batch {i//batch_size} failed: {e}")

    return all_embeddings
```

**细节 3：向量维度 vs 检索速度 vs 质量**

更高维度 ≠ 一定更好：

```
bge-small（512维）：速度最快，适合实时场景，质量略低
bge-large（1024维）：速度和质量均衡，推荐首选
text-embedding-3-large（3072维）：质量最高，存储和计算成本 3x
```

对于大多数企业知识库场景，`bge-m3`（1024 维）是最佳平衡点。

---

## 7. 在线流程 Part 1：Query 理解与增强

### 7.1 为什么 Query 需要处理

请大家想象一下这个场景：一个 Android 开发者遇到了问题，他可能会这样问：

**用户实际说的：**

> “那个处理网络请求失败的地方怎么写来着”
> **但系统理解的是：**

> “Retrofit / OkHttp 网络请求错误处理 重试机制 Kotlin”
> **为什么会有这个差距？**

用户的自然语言是"次优"的检索输入：

        * 口语化、模糊、省略关键词
        * 包含"那个"、"来着"等填充词
        * 隐含了具体的上下文（默认是Android开发）

Query 处理的**三板斧**：**把用户的自然语言提问，转换成能最大化召回率的检索输入。**

    * **意图识别**：把口语转成技术关键词
    * **查询扩展**：补充同义词、相关术语
    * **结构化**：让 query 更像“文档的语言”，提升召回率

处理后的 query 可能是：

> “Retrofit OkHttp 网络请求错误处理 重试机制 示例代码 Kotlin”
> 这样更容易匹配到相关的技术文章段落。这就像把模糊的"帮我找那本书"变成了精确的"帮我找《Java 编程思想》第 5 版 ISBN: 978-7-111-xxxxx"

### 7.2 HyDE（Hypothetical Document Embeddings）

**核心洞察**：有时候，答案比问题更接近正确答案。

**举个生动的例子：**

**用户问：** “网络请求失败怎么处理？”

**传统做法：**

        * 直接拿这个问题去搜索
        * 匹配包含"网络请求失败怎么处理"的文档

**HyDE 做法：**

**1、先让 AI 猜一个答案：**

```python
"在网络请求失败时，可以通过Retrofit的拦截器实现重试机制，
例如使用OkHttp的RetryInterceptor，设置最大重试次数和延迟策略..."
```

这个答案可能不完美，但包含大量技术关键词

**2、用这个"假设答案"去搜索：**

包含"Retrofit"、“拦截器”、“重试机制”、"OkHttp"等关键词

更容易找到真正的技术文档

**3、为什么有效？**

            * 问题可能问得很泛，但答案通常很具体
            * 假设答案包含了丰富的技术术语
            * 就像用"参考答案"去找"标准答案"

### 7.3 Query 改写与多路检索

**场景再现：** 你想学 Python 读取 CSV

**原始问题：**

> “Python 怎么读取 CSV 文件？”
> ** 步骤 1：生成多个"分身"问题**

我们用 LLM 生成 5 个不同角度的问法：

| **版本** | **问法**                                        | **特点**     |
| -------- | ----------------------------------------------- | ------------ |
| 1        | “Python pandas read_csv function usage example” | 技术精确版   |
| 2        | “How to read CSV files in Python using pandas?” | 问题形式版   |
| 3        | “Python CSV file reading code example”          | 代码导向版   |
| 4        | “用 Python 处理逗号分隔值文件的方法”            | 同义词扩展版 |
| 5        | “Python CSV 读取”                               | 简化版       |

**步骤 2：多路并行检索**

就像派出 5 个侦察兵，每个拿不同的问题去搜索：

```python
侦察兵1（技术版）→ 找到官方文档
侦察兵2（问题版）→ 找到StackOverflow回答
侦察兵3（代码版）→ 找到GitHub示例
侦察兵4（中文版）→ 找到中文博客
侦察兵5（简化版）→ 找到基础教程
```

**步骤 3：结果合并与重排**

**常用策略：**

        * **取并集**：所有找到的文档合并去重（宁多勿漏）
        * **加权融合**：给技术版的结果更高权重
        * **去重筛选**：相似内容只保留最相关的那份

**这就像：**

        * 你问5个朋友"哪家餐厅好吃"
        * 每个人推荐的不一样
        * 你把所有推荐汇总，去掉重复的
        * 最后得到一份完整的餐厅清单

### 7.4 Query 路由：**先分类，再检索**

**场景：** 一个公司有多个技术栈的知识库

**问题来了：**

> “Swift 中的 async/await 怎么用？”
> **如果没有路由：**

        * 在Android库里搜 → 无结果
        * 在前端库里搜 → 无结果
        * 在后端库里搜 → 无结果
        * 最后在iOS库里找到 → 效率低下

**路由机制：先判断"这是谁的事"**

```python
def route_query(question: str) -> str:
    """判断问题属于哪个知识库"""
    # 让AI分类：这是Android问题？iOS问题？前端问题？
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""判断以下问题属于哪个技术领域：

问题：{question}

选项：
- android（Android开发）
- ios（iOS开发）
- frontend（前端开发）
- backend（后端开发）
- general（通用问题）
"""
        }]
    )
    return response.choices[0].message.content

# 使用
domain = route_query("Swift 中的 async/await 怎么用？")
# → "ios" ✅

# 只检索iOS相关向量库
results = ios_vectorstore.similarity_search(query, k=10)
```

**路由的好处：**

        1. **精准定位**：直接找对专家
        2. **提高效率**：避免无效搜索
        3. **减少干扰**：Android文档不会干扰iOS搜索

---

## 8. 在线流程 Part 2：向量检索与混合检索

### 8.1 纯向量检索的局限

**想象这个场景：**

你在学习 Android 开发，想知道 AGP 8.0 的新特性。

**🏃‍♂️ 向量检索（语义检索）**

    * **特点**：像一位理解你"意思"的朋友
    * **优点**：能理解语义关联

```python
你问："AGP是什么？"
它能找到："Android Gradle Plugin是Android项目的构建工具..."
```

        * ✅ 理解"AGP"就是"Android Gradle Plugin"
        * ✅ “迁移"也能匹配到"升级”、"转换"相关文档

    * **缺点**：有时候"太通情达理"反而误事

```python
你问："AGP 8.0的新特性"
它可能返回：
1. AGP 7.4的文档（因为语义相似）
2. Gradle 8.0的文档（版本号相同）
3. Android Studio的更新说明（相关但不准确）
```

        * ❌ 对精确术语不敏感："AGP 8.0"和"AGP 7.4"在向量空间离得很近
        * ❌ 无法强制要求多个关键词同时出现

**🏃‍♂️ BM25（关键词检索）**

    * **特点**：像一位严格的图书管理员
    * **优点**：精确匹配，一字不差

```python
你问："AGP 8.0的新特性"
它只会返回：
1. 包含"AGP"和"8.0"的文档
2. 包含"Android Gradle Plugin"和"8.0"的文档
```

        * ✅ 必须同时出现"AGP"和"8.0"
        * ✅ 对技术名词、版本号、错误码特别有效

    * **缺点**：缺乏"变通能力"

```python
你问："如何迁移项目到新版本"
文档写的是："项目升级指南"
结果：匹配失败 ❌
```

        * ❌ 无法理解"迁移" = “升级”
        * ❌ 对同义词、近义词不敏感

### 8.2 混合检索（Hybrid Search）**：强强联合的"黄金组合"**

**混合检索 = 向量检索（语义） + BM25（词汇）两路结果合并**

| **查询类型** | **纯向量** | **纯 BM25** | **混合检索 (RRF)** | **提升幅度** |
| ------------ | ---------- | ----------- | ------------------ | ------------ |

|**通用语义查询**
（如"如何优化性能"）|0.78|0.65|0.80|+2.6%|
|**精确术语查询**
（如"K8s Pod 状态 CrashLoopBackOff"）|0.62|0.85|0.88|+41.9%|
|**混合查询**
（如"Spring Boot 3.0 的响应式编程示例"）|0.71|0.78|0.84|+18.3%|
|**平均 (Recall@10)**|0.70|0.76|0.84|+20.0|

**实测数据：** 在中文技术文档上，混合检索通常比纯向量检索提升 **Recall@10 约 8~20%**，对包含专有名词（版本号、API 名、错误码）的查询提升更明显。

**（召回）**：用向量检索从整个文档库中找出 Top-k 的候选文档。**Recall@10（召回率）：** 正确答案是否在 Top-10 结果里

---

## 9. 在线流程 Part 3：Rerank 精排，**从"海选"到"决赛"的晋级之路**

### 9.1 为什么向量检索还不够，**Android 问题的 Top-10"质量危机"**

用户问：

> “Android Studio 中如何调试内存泄漏？”
> **向量检索返回 Top-10 的结果：**

```
1. ✅ Android Profiler内存泄漏检测完整指南 - 直接相关
2. ❌ Android Studio快捷键大全 - 语义相关但不精准
3. ✅ LeakCanary框架使用教程 - 相关但工具不同
4. ❌ Android性能优化概述 - 相关但不具体
5. ✅ Java内存管理基础 - 边缘相关
6. ❌ Android UI布局优化 - 完全不相关
7. ✅ 使用MAT分析Android内存 - 部分相关
8. ❌ Gradle构建配置 - 不相关
9. ✅ Android OOM异常处理 - 相关但问题不同
10. ❌ Kotlin协程内存管理 - 完全不相关
```

**问题很明显：** 虽然第 1 个结果完美，但用户需要翻过大量"噪音"才能找到真正有用的。

向量检索的 Top-10 结果，质量参差不齐。原因：

        1. **Bi-Encoder（向量检索） 的信息损失**：Query 和 Document 分别编码，编码时无法相互「看到」对方，不能做细粒度的语义交互

```python
# Bi-Encoder在Android场景的问题
query = "Android中的Handler和Looper机制"
doc1 = "Handler消息机制详解"  # 匹配度：0.85
doc2 = "Looper工作原理分析"   # 匹配度：0.83
doc3 = "Handler导致内存泄漏的解决方案"  # 匹配度：0.79

# 实际上：
# 用户想要：Handler和Looper的协同工作机制
# 最佳文档：应该是同时包含两者的文档
# 但向量检索可能把只讲Handler的排在前面
```

        2. **余弦相似度是全局相似**：不能精细衡量「这段文字是否直接回答了这个问题」

| **Android 查询**        | **文档内容**                      | **余弦相似度** | **是否真正相关** |
| ----------------------- | --------------------------------- | -------------- | ---------------- |
| “RecyclerView 滑动卡顿” | “RecyclerView 性能优化 10 个技巧” | 0.88           | ✅ 高度相关      |
| “RecyclerView 滑动卡顿” | “Android UI 渲染原理”             | 0.82           | ❌ 相关但不直接  |
| “RecyclerView 滑动卡顿” | “ListView 使用教程”               | 0.78           | ❌ 过时技术      |
| “RecyclerView 滑动卡顿” | “Kotlin 协程优化”                 | 0.65           | ❌ 完全不相关    |

### 9.2 Cross-Encoder vs Bi-Encoder

```
Bi-Encoder（向量检索，粗排）：
  Query → Encoder → Q-Vector
  Doc   → Encoder → D-Vector
  相似度 = cosine(Q-Vector, D-Vector)

  ✅ 快：Document 向量预计算，查询只需 encode Query + 一次向量检索
  ❌ 精度有限：Query 和 Doc 编码时互相不知道对方

Cross-Encoder（Rerank，精排）：
  [Query + Doc] → Encoder → 直接输出相关性分数

  ✅ 精度高：Query 和 Doc 一起输入，Attention 机制充分交互
  ❌ 慢：每个 (Query, Doc) 对都需要一次前向计算，无法预计算
```

**为什么必须两阶段？**

```
假设你有 100 万个文档块：

如果只用 Cross-Encoder：
  100万次前向计算 × 每次 50ms = 50,000 秒 ≈ 14 小时（完全不可用）

两阶段方案：
  Bi-Encoder 粗排：100万次向量相似度 ≈ 20ms（ANN 索引加速）
  Cross-Encoder 精排：Top-50 × 50ms = 2.5 秒（可接受）
```

- **Bi-Encoder（双编码器）**：就像两个**独立的面试官**。

一个面试官（编码器 A）负责阅读并理解简历（文本 A），生成一份详细的评估报告（向量）。另一个面试官（编码器 B）阅读职位描述（文本 B），也生成一份评估报告（向量）。然后，我们只是简单地比较这两份报告（计算向量相似度，如余弦相似度）来判断匹配度。

- **Cross-Encoder（交叉编码器）**：就像**一个顶级的面试官**。

他/她同时把简历（文本 A）和职位描述（文本 B）放在一起看，仔细分析它们之间的交互、关联和矛盾，然后直接给出一个最终的匹配分数。

```python
def android_retrieval(query: str, android_docs: List[Document]):
    # 第一阶段：快速粗排
    # 从海量Android资料中筛选Top-50
    candidates = bi_encoder_search(
        query,
        android_docs,
        filter_by=["Android", "Kotlin", "Java"]  # Android特定过滤
    )
    # 耗时：~25ms（比通用快，因为有过滤）

    # 第二阶段：精确定位
    # 重点检查：版本兼容性、代码示例、解决方案
    reranked = cross_encoder_rerank(
        query,
        candidates,
        focus_on=["code_snippet", "version_info", "solution_steps"]
    )
    # 耗时：2.0s（针对Android优化）

    # 返回Top-5最适合Android开发者的
    return reranked[:5]
```

### 9.3 Rerank 对结果的影响——用数据说话

真实案例（某技术文档问答系统，测试集 200 条问题）：

- **Answer Accuracy（答案准确性）**：系统返回的答案与标准答案的匹配程度。衡量"回答是否正确"。
- **Faithfulness（忠实度/事实一致性）**：系统返回的答案是否完全基于检索到的文档内容，没有虚构或编造信息。衡量"回答是否可信"。

```
仅向量检索（Top-5）：
  Answer Accuracy: 61.5%
  Faithfulness:    72.3%

向量检索 + BGE-Reranker（Top-50 → 重排 → Top-5）：
  Answer Accuracy: 78.2%  ↑ +16.7%
  Faithfulness:    84.1%  ↑ +11.8%

向量检索 + Cohere Rerank：
  Answer Accuracy: 80.5%  ↑ +19.0%
  Faithfulness:    85.7%  ↑ +13.4%
```

**结论：加 Rerank 是性价比最高的优化手段，工程成本低，效果提升显著。**

---

## 10. 在线流程 Part 4：Prompt 构建与生成

### 10.1 RAG Prompt 的设计原则

一个好的 RAG Prompt 需要做到：

1. **告诉模型边界**：只基于检索结果回答，不能乱编
2. **注入检索结果**：结构清晰，来源可溯
3. **处理「找不到」的情况**：优雅地说「我不知道」比乱猜好得多

```python
def build_rag_prompt(
    query: str,
    retrieved_docs: list[dict],
    system_context: str = ""
) -> list[dict]:

    # 构建参考资料
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(
            f"【参考资料 {i}】\n"
            f"来源：{doc['source']} | 模块：{doc['module']}\n"
            f"内容：{doc['content']}\n"
        )
    context = "\n---\n".join(context_parts)

    system_prompt = f"""你是一个专业的技术助手，专门帮助移动端开发团队解答问题。

回答规则：
1. 只基于【参考资料】中的内容回答，不要编造参考资料中没有的信息
2. 如果参考资料中没有足够信息，明确说「根据现有文档，我无法完整回答这个问题」
3. 回答时注明信息来自哪个参考资料（用「根据参考资料X」的格式）
4. 如果涉及代码示例，保持代码的准确性
5. 使用简洁专业的技术语言

{system_context}"""

    user_prompt = f"""参考资料：
{context}

问题：{query}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
```

### 10.2 引用溯源的实现

生产级 RAG 系统必须支持答案溯源，方便用户验证：

```python
def generate_with_citations(query, retrieved_docs, llm_client):
    """生成带引用的答案"""

    prompt = build_rag_prompt(query, retrieved_docs)

    # 要求模型用特定格式标注引用
    prompt[-1]["content"] += """

请在回答中用 [ref:数字] 的格式标注引用，例如：
「Kotlin 协程中的异常处理需要使用 CoroutineExceptionHandler [ref:1]，
或者在 launch 块中使用 try-catch [ref:2]」"""

    response = llm_client.chat.completions.create(
        model="gpt-4o",
        messages=prompt,
        temperature=0.1,   # 低 temperature，减少幻觉
        max_tokens=1000
    )

    answer = response.choices[0].message.content

    # 解析引用
    import re
    citations = re.findall(r'\[ref:(\d+)\]', answer)
    cited_sources = {
        int(c): retrieved_docs[int(c)-1]['source']
        for c in set(citations)
        if int(c) <= len(retrieved_docs)
    }

    return {
        "answer": answer,
        "citations": cited_sources,
        "retrieved_docs": retrieved_docs
    }
```

### 10.3 完整的在线查询链路代码

```python
class RAGPipeline:
    def __init__(self, vectorstore, embed_model, reranker, llm_client):
        self.vectorstore = vectorstore
        self.embed_model = embed_model
        self.reranker = reranker
        self.llm = llm_client
        self.hybrid_retriever = HybridRetriever(...)

    async def query(self, user_question: str, config: dict = None) -> dict:
        config = config or {"top_k_retrieve": 50, "top_k_rerank": 5}

        # Step 1: Query 增强
        hyde_answer = await self.generate_hyde(user_question)

        # Step 2: 混合检索
        candidates = self.hybrid_retriever.retrieve(
            hyde_answer,
            top_k=config["top_k_retrieve"]
        )

        # Step 3: Rerank
        reranked = self.reranker.rerank(
            user_question,
            [doc.page_content for doc in candidates],
            top_k=config["top_k_rerank"]
        )

        # Step 4: 构建 Prompt 并生成
        result = generate_with_citations(
            user_question,
            reranked,
            self.llm
        )

        return result
```

---

## 11. RAG 效果评估体系

### 11.1 你必须有一套评估体系

**没有度量就没有优化方向。** 很多团队做了 RAG 之后，凭「感觉」说效果不错，这是不够的。

RAG 的评估分两个维度：

```
检索质量评估（Retrieval Quality）
  → 能不能找到正确的文档片段？

生成质量评估（Generation Quality）
  → 基于找到的文档，能不能给出准确且完整的回答？
```

### 11.2 检索质量指标

**Recall@K（召回率）：** 正确答案是否在 Top-K 结果里

```python
def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """检索结果的前 K 个里，有多少比例是相关文档"""
    top_k = set(retrieved_ids[:k])
    relevant_retrieved = top_k.intersection(relevant_ids)
    return len(relevant_retrieved) / len(relevant_ids)

# 示例：相关文档有 3 篇，Top-10 里找到了 2 篇
# Recall@10 = 2/3 = 0.667
```

**MRR（Mean Reciprocal Rank）：** 第一个正确结果平均排在第几位

```python
def mean_reciprocal_rank(results_list: list[list], relevant_ids_list: list[set]) -> float:
    rr_scores = []
    for retrieved, relevant in zip(results_list, relevant_ids_list):
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                rr_scores.append(1.0 / rank)
                break
        else:
            rr_scores.append(0.0)
    return sum(rr_scores) / len(rr_scores)
```

### 11.3 生成质量指标（RAGAS 框架）

RAGAS 是专门为 RAG 设计的评估框架，核心指标：

| **指标**                              | **定义**                                   | **如何计算**                                             |
| ------------------------------------- | ------------------------------------------ | -------------------------------------------------------- |
| **Faithfulness（忠实度）**            | 答案中每个声明是否都能在检索文档中找到依据 | 用 LLM 判断答案中的每个陈述是否有文档支撑                |
| **Answer Relevancy（答案相关性）**    | 答案是否真正回答了问题                     | 反向：根据答案生成多个问题，看生成的问题是否和原问题相似 |
| **Context Precision（上下文精准度）** | 检索的文档中，有多少比例是真正相关的       | 相关文档数 / 检索文档总数                                |
| **Context Recall（上下文召回率）**    | 正确回答所需的信息是否都在检索文档中       | 答案中能在文档里找到依据的比例                           |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# 准备测试数据集
test_data = {
    "question": ["iOS 如何避免循环引用？", "ANR 的常见原因是什么？"],
    "answer": ["...(模型生成的答案)...", "..."],
    "contexts": [
        ["...检索到的文档1...", "...文档2..."],   # 每个问题的检索结果
        ["...文档3...", "...文档4..."]
    ],
    "ground_truth": ["...标准答案1...", "...标准答案2..."]   # 可选
}

dataset = Dataset.from_dict(test_data)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(results)
# 输出示例：
# faithfulness         0.87
# answer_relevancy     0.82
# context_precision    0.74
# context_recall       0.79
```

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=ba45c93086d1467a9a83f176f2e1ce45&docGuid=umHNMyON7lLOVt)

### 11.4 构建测试集的最佳实践

```python
def generate_test_set_from_docs(documents, n=50):
    """用 LLM 从文档自动生成问答测试集"""
    test_cases = []

    for doc in random.sample(documents, n):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""基于以下文档内容，生成一个有价值的问题和对应的标准答案。
输出 JSON 格式：{{"question": "问题", "answer": "答案"}}

文档内容：
{doc.page_content}"""
            }]
        )

        import json
        qa = json.loads(response.choices[0].message.content)
        test_cases.append({
            "question": qa["question"],
            "ground_truth": qa["answer"],
            "source_doc_id": doc.metadata["id"]
        })

    return test_cases
```

---

## 12. LightRAG：知识图谱驱动的下一代 RAG

### 12.1 标准 RAG 的根本性缺陷

至此，我们已经把标准 RAG 的每个环节都讲透了。但有一类问题，做得再好的标准 RAG 也答不好：

**问题类型 1：多跳推理（Multi-hop Reasoning）**

```
问题：「我们的 App 在冷启动时，数据加载的完整流程是什么？」

要回答这个问题，需要串联：
  SplashActivity 的初始化逻辑（文档A）
  →  AppInitializer 的执行顺序（文档B）
  →  各个 Repository 的预加载时机（文档C）
  →  数据库初始化和网络请求的顺序关系（文档D）
  →  ...

标准 RAG：每次检索只拿到几个孤立的文档块，
没有「A调用B，B依赖C」这样的关系信息
```

**问题类型 2：全局概览**

```
问题：「我们整个 App 的模块依赖关系图是什么样的？」

标准 RAG：找不到，因为这个信息分散在几十个文档里，
没有任何一个文档块能完整回答这个问题
```

**问题类型 3：关系推理**

```
问题：「UserManager 和 AuthService 最终都依赖哪些底层组件？」

标准 RAG：找到 UserManager 相关文档，找到 AuthService 相关文档，
但不知道它们共同依赖的组件是什么——因为没有图结构
```

**本质原因：标准 RAG 是「碎片化的」知识表示，缺少实体之间的关系结构。**

### 12.2 知识图谱（Knowledge Graph）快速入门

**知识图谱用「三元组」表示知识：（主体，关系，客体）**

```
（UserManager，调用，AuthService）
（AuthService，依赖，TokenRepository）
（TokenRepository，存储在，SharedPreferences）
（UserManager，管理，User）
（User，包含，UserProfile）
（UserProfile，字段，email, userId, avatar）
```

把这些三元组连起来，就形成一张图：

```
UserManager ──[调用]──> AuthService ──[依赖]──> TokenRepository
    │                                                    │
    │[管理]                                         [存储在]
    ↓                                                    ↓
   User ──[包含]──> UserProfile              SharedPreferences
```

**图查询的魔力：** 可以做任意深度的路径查询：

```
// Cypher 查询（Neo4j 图数据库）
// 找到 UserManager 直接或间接依赖的所有组件（任意深度）
MATCH (start:Entity {name: "UserManager"})-[:依赖|调用*1..5]->(end:Entity)
RETURN end.name, end.description
ORDER BY length(path)
```

### 12.3 传统知识图谱的困境

手工构建知识图谱需要：

        * 领域专家标注实体和关系
        * 专门的 NLP 流水线（NER + 关系抽取）
        * 持续的人工维护

**成本极高，这是知识图谱长期难以规模化的根本原因。**

### 12.4 LightRAG 的核心创新

**LightRAG（2024，香港大学，arxiv: 2410.05779）的核心思路：**

> **用 LLM 代替人工，自动从非结构化文档中抽取实体和关系，构建知识图谱；然后将图检索和向量检索结合，支持局部和全局两种查询模式。**
> **三大核心创新：**

        1. **LLM 驱动的图构建**：无需人工标注，LLM 自动完成实体抽取和关系识别
        2. **双层检索（Local + Global）**：根据问题类型选择不同的检索策略
        3. **增量更新**：新文档进来只更新图的局部，不需要重建全量

### 12.5 LightRAG 图构建流程详解

#### 阶段 1：实体和关系抽取，**生成扁平实体图**

```python
# LightRAG 的核心抽取 Prompt（简化版）
EXTRACT_PROMPT = """
你是一个知识图谱构建专家。从以下文本中抽取：
1. 实体（Entity）：技术组件、类名、模块、概念等
2. 关系（Relation）：实体之间的相互关系

严格按照以下 JSON 格式输出：
{
  "entities": [
    {
      "name": "实体名称",
      "type": "CLASS/MODULE/CONCEPT/API/PATTERN",
      "description": "实体的详细描述，包含其功能和用途"
    }
  ],
  "relations": [
    {
      "source": "源实体名称",
      "target": "目标实体名称",
      "relation": "关系动词（如：调用、依赖、继承、实现、管理）",
      "description": "关系的详细描述",
      "weight": 0.8  // 关系强度 0~1
    }
  ]
}

文本内容：
{chunk_text}
"""
```

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=af7435dcac934d66b40ad6330cbadee3&docGuid=umHNMyON7lLOVt)

**抽取示例：**

```
输入文本：
"UserViewModel 继承自 BaseViewModel，通过注入 UserRepository
来获取用户数据。当调用 loadUser(userId) 时，它会先检查内存缓存，
缓存未命中则通过 UserRepository.getUser() 发起网络请求。
网络请求由 RetrofitService 处理，结果通过 Flow 返回。"

抽取结果：
{
  "entities": [
    {"name": "UserViewModel", "type": "CLASS", "description": "用户数据的 ViewModel，管理用户信息的 UI 状态"},
    {"name": "BaseViewModel", "type": "CLASS", "description": "所有 ViewModel 的基类，提供公共生命周期处理"},
    {"name": "UserRepository", "type": "CLASS", "description": "用户数据仓库，协调内存缓存和网络数据源"},
    {"name": "RetrofitService", "type": "API", "description": "封装 HTTP 网络请求的 Retrofit 接口"},
    {"name": "Flow", "type": "CONCEPT", "description": "Kotlin 异步数据流，用于响应式数据传递"}
  ],
  "relations": [
    {"source": "UserViewModel", "target": "BaseViewModel", "relation": "继承", "weight": 1.0},
    {"source": "UserViewModel", "target": "UserRepository", "relation": "依赖注入", "weight": 0.9},
    {"source": "UserRepository", "target": "RetrofitService", "relation": "调用", "weight": 0.8},
    {"source": "RetrofitService", "target": "Flow", "relation": "返回", "weight": 0.7}
  ]
}
```

#### 阶段 2：实体向量化和图存储，**图索引 + 向量索引“双轨”**

- **双轨并行**：

  - **向量索引**：为每个文本片段（chunk）生成向量，提供**快速、全局的语义检索**。
  - **图索引**：存储实体和关系，提供**精确、结构化的关联检索**。

- **检索协同**：

  - **查询时**：向量检索引擎快速找到相关文本块（粗筛）。
  - **同时/随后**：从查询和返回的文本块中提取实体，在图索引中进行**图遍历**（如查找邻居、多跳关系），找到与查询强相关的结构化信息。

- **这正是 LightRAG 等方案的核心**：结合了**语义广度**（向量检索）和**逻辑深度**（图谱检索）。

#### 阶段 3：**增量并集合并，新增文档只追加**

- **增量构建**：这是工业级应用的关键。系统可以持续学习新文档，而无需从头重建整个索引。
- **操作方式**：

  - 为新文档抽取实体和关系。
  - 将新节点/边与现有图谱进行**并集合并**。通常需要一个**实体消歧/对齐**步骤（判断新文档中的“苹果”是否与已有图谱中的“苹果”公司是同一实体）。
  - 为新文本片段生成向量，**直接追加**到现有向量索引中。

- **优势**：
  - 效率极高，支持实时或准实时更新。
  - 资源消耗低，只需处理新增数据。

### 12.6 检索模式

| **Mode**   | **用图谱** | **检索策略**                                      | **上下文范围**                                             | **适用场景**                 | **优点**                   | **缺点**                               |
| ---------- | ---------- | ------------------------------------------------- | ---------------------------------------------------------- | ---------------------------- | -------------------------- | -------------------------------------- |
| **naive**  | 否         | **纯向量**检索                                    | 命中文本块                                                 | 无图谱/快速问答              | 实现简单，延迟低           | 无法利用结构化关系，跨段落能力弱       |
| **local**  | 是         | 向量/关键词定位后，在图谱中做**局部邻域检索**     | ll_keywords **\_get_node_data**相关节点及其 k-hop 邻域     | 追溯实体/事件脉络            | 能抓上下游关系，语境更丰富 | 覆盖范围有限，可能漏掉远距离信息       |
| **global** | 是         | **全局**图谱检索/聚合                             | hl_keywords **\_get_edge_data**全图级别                    | 全局主题、趋势、跨章节分析   | 可整合全局信息，宏观视角强 | 延迟高，token 消耗大                   |
| **hybrid** | 是         | **向量检索** + **图谱**检索结果融合               | ll_keywords + hl_keywords 命中文本块 + 相关子图            | 既要语义相似证据又要结构脉络 | 准确性高，鲁棒性好         | 检索/融合复杂，延迟较高                |
| **mix**    | 是         | 与 hybrid 类似，但融合策略更激进（权重/顺序不同） | ll_keywords + hl_keywords+chunks_vdb 命中文本块 + 相关子图 | 需要更强融合/探索性检索      | 可能在部分场景表现更优     | 效果依赖实现细节，稳定性不一定最好     |
| **bypass** | 否         | **不检索**，直接 LLM 推理                         | 无                                                         | 纯对话、创作、工具测试       | 延迟最低，逻辑简单         | 无外部知识，依赖模型记忆，事实可靠性差 |

> 引用来源：[LightRAG](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/nrXzHm2JzJ/oK9WrgAUacuUxS#anchor-712fac10-767b-11f0-a85c-0b10cba383ed?t=mention&mt=doc&dt=doc)

### 12.7 LightRAG vs 标准 RAG：全面对比

| **对比维度**   | **标准 RAG**           | **LightRAG**                        |
| -------------- | ---------------------- | ----------------------------------- |
| **知识表示**   | 文档块（Fragment）     | 实体 + 关系图（Knowledge Graph）    |
| **擅长问题**   | 精确问答、关键词匹配   | 关系推理、多跳推理、全局概览        |
| **构建成本**   | 低（只需 Embedding）   | 高（需要 LLM 做抽取，Token 消耗大） |
| **查询延迟**   | 低（向量检索 + 生成）  | 中（图遍历 + 向量 + 生成）          |
| **可解释性**   | 中（知道来自哪个文档） | 高（知道是哪个实体的哪条关系）      |
| **知识更新**   | 增量简单（插入新块）   | 增量中等（更新节点和边）            |
| **适合数据量** | 任意规模               | 中等规模（图太大时开销大）          |
| **适合团队**   | 入门首选               | 有一定 RAG 经验，需要关系查询       |

**什么时候必须用 LightRAG？**

- 需要回答「A 和 B 的关系」类型的问题
- 需要做代码库、架构的全局分析
- 知识库里实体间关系复杂（如 SDK 依赖、系统架构）
- 需要多跳推理（「X 影响 Y，Y 影响 Z，X 对 Z 的影响是什么」）

---

## 13 一张图总结整个体系

```
原始文档
    │
    ▼
[预处理] 去噪、清洗、结构化、提取 Metadata
    │
    ▼
[切块] Recursive / Parent-Child / Structure-aware
    │
    ├──────────────────────────┐
    ▼                          ▼
[Embedding]              [LLM 实体抽取]
bge-m3 向量化            LightRAG 图构建
    │                          │
    ▼                          ▼
[向量数据库]            [知识图谱]
Qdrant                  NetworkX / Neo4j
    │                          │
    └──────────┬───────────────┘
               │
         用户查询进来
               │
               ▼
         [Query 处理]
         HyDE / 多路改写
               │
               ▼
    ┌──────────┴──────────┐
    ▼                     ▼
[向量检索]          [BM25检索]    ← 混合检索
Top-50              Top-50
    └──────────┬──────────┘
               │ RRF 融合
               ▼
          [Rerank]
     Cross-Encoder 精排
          Top-5
               │
               ▼
        [Prompt 构建]
     问题 + 检索结果 + 系统提示
               │
               ▼
           [LLM 生成]
               │
               ▼
        答案 + 来源引用
               │
               ▼
       [记录日志 + 反馈]
         持续优化闭环
```

---

## 14 延伸阅读

| **资源**                                                                                                    | **类型** | **说明**                             |
| ----------------------------------------------------------------------------------------------------------- | -------- | ------------------------------------ |
| [LightRAG 论文](https://arxiv.org/abs/2410.05779)                                                           | 论文     | 原始论文，Hong Kong University，2024 |
| [LightRAG GitHub](https://github.com/HKUDS/LightRAG)                                                        | 代码     | 官方实现，活跃维护                   |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172)                                                      | 论文     | 揭示长 Context 中间信息丢失          |
| [RAGAS](https://github.com/explodinggradients/ragas)                                                        | 框架     | RAG 效果评估标准框架                 |
| [BGE 系列模型](https://huggingface.co/BAAI)                                                                 | 模型     | 中文最强 Embedding + Rerank          |
| [Qdrant 文档](https://qdrant.tech/documentation/)                                                           | 文档     | 生产级向量数据库推荐                 |
| [Advanced RAG](https://towardsdatascience.com/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6) | 博客     | 进阶 RAG 技术图解总览                |

---

> **写在最后**
> RAG 的本质，是把「通用智能（LLM）」和「领域知识（你的文档）」高效结合的桥梁。
> 做好这座桥，70% 取决于你怎么整理和检索知识，30% 才是模型本身。
> LightRAG 把这座桥从「高速公路」升级成了「立体交通网」——不只是点对点的文本匹配，
> 而是可以在知识的图谱里自由游走，找到任意实体之间的关系路径。
> 从今天开始，你已经不只是一个使用 AI 工具的开发者，
> 而是能理解并构建 AI 知识系统的工程师。

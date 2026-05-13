"""Multi-hop 路由触发判定（`supervisor` / `query_classifier` 在 PR-3 会调用）。

判定顺序（与 `docs/phase3-multi-hop-rag.md` §2.3 一致）：
1. 预处理 query：strip、全角标点归一（？→ ""、？? → ""）、多余空格折叠。
2. **先跑 `MULTI_HOP_NEGATIVE_GATES`**：任一命中 → 强制单跳，函数返回 `False`。
   这样 "WAI-ARIA 和虚拟列表有什么区别？" 这类简单二元对比不会吃 decompose 税。
3. 再跑 `MULTI_HOP_TRIGGERS`：任一命中 → 进入 multi-hop，函数返回 `True`。
4. 都未命中 → 返回 `False`，由上层决定要不要走 LLM 兜底分类。

本模块只提供纯函数 + 预处理 + 正则匹配，不依赖 LangGraph 或 LLM，便于单测。
"""

from __future__ import annotations

import re

from app.constants.multi_hop import (
    MULTI_HOP_NEGATIVE_GATES,
    MULTI_HOP_TRIGGERS,
)

# 预编译正则：模块加载一次即可；大小写不敏感以兼容 "vs/VS/Vs" 等混写。
_NEGATIVE_GATE_PATTERNS = tuple(
    re.compile(pat, re.IGNORECASE) for pat in MULTI_HOP_NEGATIVE_GATES
)
_TRIGGER_PATTERNS = tuple(
    re.compile(pat, re.IGNORECASE) for pat in MULTI_HOP_TRIGGERS
)

# 需要在 gate 入口归一掉的末尾标点（避免 "$" 锚点被问号绊倒）。
_TRAILING_PUNCT = ("?", "？", "。", ".", "!", "！", "～", "~", "、", ",", "，")


def preprocess_query_for_gate(query: str) -> str:
    """把 query 归一到 gate 正则假设的形态。

    步骤：
    1. strip 首尾空白；
    2. 折叠中间连续空白为单空格（避免 `"X   和   Y"` 误判）；
    3. 剥离末尾标点（中英文问号 / 句号等），让 `$` 锚点稳定匹配；
    4. 不做大小写归一 —— 交给正则 `re.IGNORECASE` 处理。
    """

    text = (query or "").strip()
    if not text:
        return ""
    # 折叠空白
    text = re.sub(r"\s+", " ", text)
    # 剥离末尾标点（支持连写 "???"）
    while text and text[-1] in _TRAILING_PUNCT:
        text = text[:-1].rstrip()
    return text


def matches_negative_gate(query: str) -> bool:
    """query 是否命中任一 negative gate（simple comparison / 定义类）。"""

    normalized = preprocess_query_for_gate(query)
    if not normalized:
        # 空 query 不走 multi-hop，算作"被拦下"。
        return True
    return any(pat.search(normalized) for pat in _NEGATIVE_GATE_PATTERNS)


def matches_positive_trigger(query: str) -> bool:
    """query 是否命中任一正向触发正则（跨文档链式 / 方案生成等）。"""

    normalized = preprocess_query_for_gate(query)
    if not normalized:
        return False
    return any(pat.search(normalized) for pat in _TRIGGER_PATTERNS)


def should_enter_multi_hop(query: str) -> bool:
    """综合判定：negative gate 优先于 positive trigger。

    返回值：
    - True  → 进入 multi-hop 流程；
    - False → 不进入（supervisor 决定走单跳 RAG 或继续 LLM 兜底分类）。
    """

    if matches_negative_gate(query):
        return False
    return matches_positive_trigger(query)


__all__ = [
    "matches_negative_gate",
    "matches_positive_trigger",
    "preprocess_query_for_gate",
    "should_enter_multi_hop",
]

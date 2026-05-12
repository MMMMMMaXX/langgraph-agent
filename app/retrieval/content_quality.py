"""文档内容质量判定：检测"模板占位符骨架"类型的低质量 chunk。

背景：
- 知识库里有"写作指南"/"skills 构建指南"这类模板说明文档，其中大量使用
  `[常见错误消息]` / `[原因：为什么会出现]` / `[怎么修]` / `[卡片ID]` 这样的
  占位符示范，原文本身对读者有意义（展示模板骨架）。
- 但作为 RAG 检索语料被命中后，LLM 会照抄这些占位符生成答案，例如 eval 曾
  观察到："根据知识库资料[1]，如果验证失败，常见原因有[原因：为什么会出现]，
  但资料未明确说明..."。这类 chunk 无法支撑事实性问答。
- ingestion 阶段暂不强制过滤（避免把模板文档整体排除），检索阶段按启发式
  丢弃这类 chunk，保证下游 LLM 拿到的只是"有实际信息"的片段。

Citation refs（`[1]` / `[2]`）只含数字，不会误伤。
"""

from __future__ import annotations

import re

# 匹配 1~24 字符的方括号标注；24 是经验上限，过长往往是转义或误写。
_PLACEHOLDER_RE = re.compile(r"\[([^\[\]]{1,24})\]")

# 允许的白名单 bracket 内容：纯数字（citation refs）、数字+字母组合（如 v1）。
# 这类内容即便多次出现也不认为是"模板骨架"。
_WHITELIST_RE = re.compile(r"^[0-9A-Za-z._\-]+$")

# 至少出现 N 个"可疑占位符"才判定为模板骨架；1 个不算，避免误伤正常对话
# 里偶尔出现的括号修辞。
_PLACEHOLDER_THRESHOLD = 2


def _is_suspicious_placeholder(bracket_content: str) -> bool:
    """判定一个 `[...]` 片段是否"像"模板占位符。

    判定规则（保守优先）：
    - 内容非空、非纯 ASCII 标识（白名单）→ 可疑；
    - 内容含冒号/中文（如 `原因：为什么会出现`）→ 更可疑，但这里统一用
      "非白名单"这一条即可覆盖，不再加额外分支。
    """

    text = bracket_content.strip()
    if not text:
        return False
    if _WHITELIST_RE.fullmatch(text):
        return False
    return True


def looks_like_template_placeholder(content: str) -> bool:
    """判断 chunk 内容是否以模板占位符为主骨架。

    阈值逻辑有意简单：出现 ≥ 2 个可疑占位符即判为骨架。实际在 eval 中观察到的
    案例（`## 故障排查 错误：[常见错误消息] 原因：[为什么会出现] 解决方法：[怎么修]`）
    一次有 3 个，裕度足够。
    """

    if not content:
        return False
    suspicious = sum(
        1 for m in _PLACEHOLDER_RE.finditer(content) if _is_suspicious_placeholder(m.group(1))
    )
    return suspicious >= _PLACEHOLDER_THRESHOLD


__all__ = ["looks_like_template_placeholder"]

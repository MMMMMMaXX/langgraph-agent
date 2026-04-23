"""纯文本解析/整理工具。

这些函数的共同特点：只操作字符串 / dict，不依赖 NovelScriptState、
不调用 LLM、不读配置。因此它们最容易写单元测试，也是重构时最安全的一部分。

拆到独立模块有两个好处：
1. 把 agent 流程逻辑和“脚本格式细节”解耦——未来剧本格式改版，只改这里
2. finalizer / tool_dispatch / review 等多个模块都依赖它们，集中放置避免循环依赖
"""

from __future__ import annotations

import re


def extract_scene_script_text(scene_draft: dict) -> str:
    """从 scene_draft 中取出剧本正文。"""

    return (scene_draft.get("script") or "").strip()


def extract_named_field(field_name: str, scene_script: str) -> str:
    """从 markdown 风格剧本正文里抽取“xxx：内容”单行字段。"""

    patterns = [
        rf"\*\*{re.escape(field_name)}：\*\*(.+)",
        rf"{re.escape(field_name)}：(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, scene_script)
        if match:
            return match.group(1).strip()
    return ""


def extract_bullet_values(section_title: str, scene_script: str) -> list[str]:
    """抽取“**xxx：**\\n- a\\n- b\\n”这种列表段里的条目。

    会清掉条目里的“：附注”和“（注释）”等尾巴，只留主名称。
    """

    pattern = rf"\*\*{re.escape(section_title)}：\*\*\n((?:- .+\n?)*)"
    match = re.search(pattern, scene_script)
    if not match:
        return []
    block = match.group(1).strip()
    if not block:
        return []
    values: list[str] = []
    for line in block.splitlines():
        cleaned = line.strip()
        if not cleaned.startswith("- "):
            continue
        value = cleaned[2:].strip()
        if "：" in value:
            value = value.split("：", 1)[0].strip()
        if "（" in value:
            value = value.split("（", 1)[0].strip()
        values.append(value)
    return values


def remove_duplicate_scene_headers(scene_script: str) -> str:
    """去掉 scene_draft 正文里重复的“场景ID/摘要/剧本正文”标题行。

    finalizer 组装多场景时会自己打剧情段落头，不需要每段再带一次。
    """

    lines = scene_script.splitlines()
    filtered: list[str] = []
    prefixes = ("**场景ID：", "**场景摘要：", "**剧本正文：**")
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in prefixes):
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()


def dedupe_keep_order(items: list[str]) -> list[str]:
    """按首次出现顺序去重，忽略空白项。"""

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result

"""`novel_script_agent` 测试。

分两层：
1. **pure-compute（5 条）**：`looks_like_script_task` / `extract_source_text` /
   `count_chapter_boundaries` / `calculate_max_iterations`。全是字符串/正则/算术，
   无外部依赖，稳而快。
2. **node 编排（3 条）**：`novel_script_agent_node` 内部会去 invoke 一整个 ReAct
   子图（`novel_script_graph.invoke`），真实跑链路会触发 N 次 LLM + tool。测试里
   monkeypatch `novel_script_graph.invoke` 返回一个可控的 dict，验证 node 层
   的"映射主 AgentState → NovelScriptState → 再组装 next_state"这段逻辑。

不去测子图内部的理由：graph 内部有 planner/write/review 三个 LLM 调用 + tool
dispatch，深度测试应该在 novel_script/* 子模块里单独做，放在 node 层会糊成一团。
"""

from __future__ import annotations

from app.agents.novel_script.constants import (
    DEFAULT_TARGET_SCENE_COUNT,
    FAST_MODE_SOURCE_CHARS,
    FAST_MODE_TARGET_SCENE_COUNT,
    MAX_SCENE_REWRITE_ATTEMPTS,
    NOVEL_SCRIPT_NO_REVIEW_FINALIZE_BUFFER,
    NOVEL_SCRIPT_REVIEW_CYCLE_ITERATIONS,
    NOVEL_SCRIPT_SETUP_ITERATIONS,
)
from app.agents.novel_script_agent import (
    calculate_max_iterations,
    count_chapter_boundaries,
    extract_source_text,
    looks_like_script_task,
    novel_script_agent_node,
)
from app.constants.routes import ROUTE_NOVEL_SCRIPT_AGENT

# ------------------------------- pure-compute -------------------------------


def test_looks_like_script_task_hits_keywords() -> None:
    assert looks_like_script_task("把这段改成剧本")
    assert looks_like_script_task("请改编成短剧")
    assert looks_like_script_task("帮我生成剧本")
    # 否定样例：普通问答不该被误判为创作任务
    assert not looks_like_script_task("剧本是什么意思")
    assert not looks_like_script_task("今天天气怎么样")


def test_extract_source_text_prefers_colon_separator() -> None:
    # 常见模式："把下面这段改成剧本：<正文>" → 冒号后是正文
    msg = "把下面这段改成剧本：从前有座山，山里有座庙。"
    assert extract_source_text(msg) == "从前有座山，山里有座庙。"


def test_extract_source_text_falls_back_to_blank_line() -> None:
    # 没有冒号时，空行也是有效分隔
    msg = "请把它改成剧本\n\n第一段正文内容"
    assert extract_source_text(msg) == "第一段正文内容"


def test_extract_source_text_returns_whole_when_no_separator() -> None:
    # 纯正文（没有指令头）直接原样返回
    msg = "从前有座山"
    assert extract_source_text(msg) == "从前有座山"


def test_count_chapter_boundaries_recognizes_chinese_and_arabic() -> None:
    text = """第1章 开端
前情提要一些文字
第二章 转折
更多内容
第三回 高潮
最后收尾"""
    # 阿拉伯数字 + 中文数字 + 回/章 都要支持
    assert count_chapter_boundaries(text) == 3

    # 没有章节标题 → 0，不可误算
    assert count_chapter_boundaries("普通一段文字，没有任何章节标题。") == 0


def test_calculate_max_iterations_review_enabled_formula() -> None:
    # enable_review=True: setup + target + target*rewrite_attempts + review_cycle
    target = 3
    expected = (
        NOVEL_SCRIPT_SETUP_ITERATIONS
        + target
        + target * MAX_SCENE_REWRITE_ATTEMPTS
        + NOVEL_SCRIPT_REVIEW_CYCLE_ITERATIONS
    )
    assert calculate_max_iterations(target, enable_review=True) == expected


def test_calculate_max_iterations_review_disabled_formula() -> None:
    # enable_review=False: setup + target + finalize_buffer
    target = 3
    expected = (
        NOVEL_SCRIPT_SETUP_ITERATIONS + target + NOVEL_SCRIPT_NO_REVIEW_FINALIZE_BUFFER
    )
    assert calculate_max_iterations(target, enable_review=False) == expected
    # review=False 必定比 review=True 少：少掉 target*rewrite + review_cycle - buffer
    assert calculate_max_iterations(target, False) < calculate_max_iterations(
        target, True
    )


# ----------------------------- novel_script_agent_node -----------------------------


def _patch_graph(monkeypatch, fake_result: dict) -> dict:
    """把 novel_script_graph.invoke 打桩成固定返回。

    额外记录一次 invoke 的入参 react_state，供测试断言主 state → 子图 state 的映射。
    """
    import app.agents.novel_script_agent as node_mod

    captured: dict = {"react_state": None}

    class FakeGraph:
        def invoke(self, react_state):
            captured["react_state"] = react_state
            return fake_result

    monkeypatch.setattr(node_mod, "novel_script_graph", FakeGraph())
    return captured


def _state(message: str) -> dict:
    return {"messages": [{"role": "user", "content": message}]}


def test_node_short_text_uses_default_target_scene_count(monkeypatch) -> None:
    captured = _patch_graph(
        monkeypatch,
        fake_result={
            "final_script": "SCENE 1\n...\nSCENE 2\n...\nSCENE 3\n...",
            "scene_plan": [{"id": 1}, {"id": 2}, {"id": 3}],
            "scene_drafts": [{"id": 1}, {"id": 2}, {"id": 3}],
            "review_notes": [],
            "iteration_count": 8,
        },
    )

    # 短文本（< FAST_MODE_SOURCE_CHARS），无章节 → 用 DEFAULT_TARGET_SCENE_COUNT
    short = "一段很短的小说。" * 5  # 远小于 2500 字
    result = novel_script_agent_node(_state(f"改成剧本：{short}"))

    # 映射结果：子图 state 里 target_scene_count 走默认值
    assert captured["react_state"]["target_scene_count"] == DEFAULT_TARGET_SCENE_COUNT
    assert captured["react_state"]["fast_mode"] is False
    assert captured["react_state"]["enable_review"] is True

    # 主 state 输出结构
    assert result["answer"].startswith("SCENE")
    assert result["agent_outputs"][ROUTE_NOVEL_SCRIPT_AGENT] == result["answer"]
    debug = result["debug_info"][ROUTE_NOVEL_SCRIPT_AGENT]
    assert debug["scene_count"] == 3
    assert debug["draft_count"] == 3
    assert debug["iteration_count"] == 8


def test_node_long_text_enters_fast_mode(monkeypatch) -> None:
    captured = _patch_graph(
        monkeypatch,
        fake_result={
            "final_script": "FAST SCRIPT",
            "scene_plan": [{"id": 1}, {"id": 2}],
            "scene_drafts": [{"id": 1}, {"id": 2}],
            "review_notes": [],
        },
    )

    # 大于 FAST_MODE_SOURCE_CHARS(2500) 且大于 REVIEW_ENABLED_MAX_SOURCE_CHARS(4500)：
    # fast_mode 开启 + target 降到 2 + review 关闭（省成本）
    long_text = "小说正文。" * 1000  # 5000 字，同时越过两个阈值
    result = novel_script_agent_node(_state(f"改成剧本：{long_text}"))

    assert captured["react_state"]["fast_mode"] is True
    assert captured["react_state"]["target_scene_count"] == FAST_MODE_TARGET_SCENE_COUNT
    assert captured["react_state"]["enable_review"] is False

    assert result["answer"] == "FAST SCRIPT"
    assert result["debug_info"][ROUTE_NOVEL_SCRIPT_AGENT]["source_text_len"] > (
        FAST_MODE_SOURCE_CHARS
    )


def test_node_chapter_count_overrides_default_target(monkeypatch) -> None:
    captured = _patch_graph(
        monkeypatch,
        fake_result={
            "final_script": "CHAPTERS",
            "scene_plan": [{"id": i} for i in range(5)],
            "scene_drafts": [{"id": i} for i in range(5)],
        },
    )

    # 明确含 5 个章节标题 → 尊重作者切分，target=5，不走 default=3
    text = """改成剧本：
第一章 序
内容A
第二章 发展
内容B
第3章 转折
内容C
第四章 高潮
内容D
第五章 结尾
内容E"""
    novel_script_agent_node(_state(text))

    assert captured["react_state"]["target_scene_count"] == 5


def test_node_empty_result_falls_back_to_insufficient(monkeypatch) -> None:
    # 子图啥也没产出 → answer 兜底成 "资料不足"，不应崩
    _patch_graph(
        monkeypatch,
        fake_result={"final_script": ""},
    )

    result = novel_script_agent_node(_state("改成剧本：一段文字"))
    assert result["answer"] == "资料不足"

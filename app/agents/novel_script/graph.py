"""兼容门面：将 novel_script ReAct 图入口转发到 `app.agents.novel_script.react`。

历史上 `novel_script_agent.py` 等调用方直接
`from app.agents.novel_script.graph import novel_script_graph`。
拆分后 graph 实际构建逻辑已经移入 `react/builder.py`，
这里保留最小 re-export，避免修改已有调用点。
"""

from __future__ import annotations

from app.agents.novel_script.react import (  # noqa: F401  - re-export
    build_novel_script_graph,
    novel_script_graph,
)

__all__ = ["build_novel_script_graph", "novel_script_graph"]

"""FastAPI 应用实例 + 系统级路由（/health、/debug-ui）。

业务路由放在 routes.py 里，这里只负责：
1. 创建 FastAPI 实例（供 uvicorn / TestClient 使用）
2. 挂上系统类路由
3. 把业务 router include 进来
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from .routes import router as chat_router

# 调试页面在仓库内随包分发，不依赖运行目录。
DEBUG_UI_PATH = Path(__file__).resolve().parent.parent / "debug_ui.html"

app = FastAPI(title="LangGraph Agent API")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug-ui")
def debug_ui() -> FileResponse:
    return FileResponse(DEBUG_UI_PATH)


app.include_router(chat_router)

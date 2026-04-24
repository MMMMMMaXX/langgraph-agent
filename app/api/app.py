"""FastAPI 应用实例 + 系统级路由（/health、/health/ready、/debug-ui）。

业务路由放在 routes.py 里，这里只负责：
1. 创建 FastAPI 实例（供 uvicorn / TestClient 使用）
2. 挂上系统类路由
3. 把业务 router include 进来
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from app.config import CONVERSATION_HISTORY_CONFIG, VECTOR_STORE_CONFIG
from app.constants.model_profiles import PROFILE_DEFAULT_CHAT
from app.llm.providers import get_profile_runtime_info

from .routes import router as chat_router

# 调试页面在仓库内随包分发，不依赖运行目录。
DEBUG_UI_PATH = Path(__file__).resolve().parent.parent / "debug_ui.html"

app = FastAPI(title="LangGraph Agent API")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Liveness 探针：只要进程活着就返回 ok。

    用于容器/编排系统判断是否需要强制重启。
    不检查任何外部依赖，避免"依赖抖动导致被重启"。

    同时回带镜像级版本信息（由 Dockerfile 构建时注入），方便：
    - 确认线上跑的是哪个 commit / 哪个版本
    - 灰度 / 回滚验证
    - 监控按 version 维度聚合
    """

    return {
        "status": "ok",
        "version": os.getenv("APP_VERSION", "dev"),
        "git_sha": os.getenv("APP_GIT_SHA", "unknown"),
        "build_time": os.getenv("APP_BUILD_TIME", "unknown"),
    }


def _check_llm_profile() -> tuple[bool, dict[str, str]]:
    """确认默认 chat profile 能解析到可用的 provider + model + api_key。"""

    try:
        info = get_profile_runtime_info(PROFILE_DEFAULT_CHAT, kind="chat")
    except Exception as exc:  # noqa: BLE001
        return False, {"profile": PROFILE_DEFAULT_CHAT, "error": str(exc)}

    if not info.get("model"):
        return False, {**info, "error": "model empty"}

    # provider 对应的 api_key 是否存在（通过 ENV 反查）。
    provider = info.get("provider", "")
    api_key_env = f"{provider.upper()}_API_KEY" if provider else ""
    if api_key_env and not os.getenv(api_key_env, "").strip():
        return False, {**info, "error": f"{api_key_env} not set"}

    return True, info


def _check_history_path() -> tuple[bool, dict[str, str]]:
    """确认会话历史后端对应的目录可写。"""

    cfg = CONVERSATION_HISTORY_CONFIG
    path = cfg.sqlite_path if cfg.backend == "sqlite" else cfg.jsonl_path
    parent = Path(path).resolve().parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, {"backend": cfg.backend, "path": path, "error": str(exc)}

    if not os.access(parent, os.W_OK):
        return False, {
            "backend": cfg.backend,
            "path": path,
            "error": "parent not writable",
        }

    return True, {"backend": cfg.backend, "path": path}


def _check_chroma_dir() -> tuple[bool, dict[str, str]]:
    """确认 Chroma 持久化目录可创建 / 可写。"""

    persist_dir = Path(VECTOR_STORE_CONFIG.chroma_persist_dir).resolve()
    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, {"persist_dir": str(persist_dir), "error": str(exc)}

    if not os.access(persist_dir, os.W_OK):
        return False, {"persist_dir": str(persist_dir), "error": "not writable"}

    return True, {"persist_dir": str(persist_dir)}


@app.get("/health/ready")
def health_ready() -> JSONResponse:
    """Readiness 探针：依赖齐备才返回 200。

    任一检查项失败都会返回 503，用于：
    - 部署时确认新版本真的能工作再切流量
    - 负载均衡从健康池中摘除异常实例
    - 快速暴露"配置写错但进程启动了"这类隐性问题
    """

    checks: dict[str, dict[str, object]] = {}
    all_ok = True

    for name, checker in (
        ("llm_profile", _check_llm_profile),
        ("history_path", _check_history_path),
        ("chroma_dir", _check_chroma_dir),
    ):
        ok, detail = checker()
        checks[name] = {"ok": ok, **detail}
        if not ok:
            all_ok = False

    payload = {"status": "ready" if all_ok else "not_ready", "checks": checks}
    return JSONResponse(payload, status_code=200 if all_ok else 503)


@app.get("/debug-ui")
def debug_ui() -> FileResponse:
    return FileResponse(DEBUG_UI_PATH)


app.include_router(chat_router)

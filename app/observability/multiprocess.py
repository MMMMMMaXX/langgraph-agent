"""Prometheus multiproc dir 生命周期管理（PR-1）。

监控方案 §9 PR-1 要求多 worker 安全的四件套：
1. **启动前清空** `PROMETHEUS_MULTIPROC_DIR`，避免上次崩溃残留 db 让计数翻倍
   （兜底主路径——SIGKILL/OOM 不会跑任何退出钩子）；
2. **优雅退出** 调用 `mark_process_dead(pid)` 回收 worker 样本；
3. **stale pid 巡检** 周期任务清理 pid 已不存在的 db，覆盖崩溃残留；
4. **dev/reload 模式** fail-fast：reloader fork 出的子进程 pid 复用会污染样本，
   不允许同时启用 `--reload` 与 multiproc backend。

本模块只提供工具函数，调度由 FastAPI lifespan 决定。
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

from prometheus_client import multiprocess

from app.constants.metrics import (
    METRIC_OBSERVABILITY_MULTIPROC_DIR_ANOMALY_TOTAL,
)
from app.observability.emit import emit_counter
from app.utils.logger import logger

# 环境变量：开启 multiproc backend 的开关，由 prometheus_client 原生识别。
ENV_PROMETHEUS_MULTIPROC_DIR = "PROMETHEUS_MULTIPROC_DIR"

# 环境变量：显式声明当前进程允许同时启用 multiproc + reload（仅用于绕开 fail-fast 的高级场景）。
ENV_OBS_ALLOW_RELOAD_MULTIPROC = "OBS_ALLOW_RELOAD_MULTIPROC"

# 环境变量：显式允许 lifespan 在 multiproc 模式下也兜底清理 multiproc dir。
# 默认关闭——多 worker 时每个 worker 的 lifespan 都会触发清理，
# 后启动的 worker 会把先启动 worker 的样本文件擦掉，造成样本黑洞。
# 仅在"单进程本地裸跑 + 想让 app 自己清"的场景下可置 1。
ENV_OBS_LIFESPAN_PREPARE_MULTIPROC_DIR = "OBS_LIFESPAN_PREPARE_MULTIPROC_DIR"

# stale pid 巡检默认周期（秒）。
DEFAULT_SCAN_INTERVAL_SECONDS = 30.0

# 只有 `gauge_livesum_<pid>.db` / `gauge_liveall_<pid>.db` 这两类文件
# 会被 prometheus_client.multiprocess.mark_process_dead(pid) 清理。
# Counter / Histogram 以及其它 gauge 模式（min/max/all/mostrecent）的样本
# 即使 pid 已死也必须保留——MultiProcessCollector 会按规则继续聚合。
# 因此 stale pid 巡检只关注 live* 类型，以免把不该清的文件误判为异常。
_LIVE_GAUGE_FILE = re.compile(r"^gauge_live(sum|all)_(\d+)\.db$")


def get_multiproc_dir() -> str | None:
    """返回当前进程使用的 multiproc dir；未配置返回 None（视为单进程模式）。"""

    value = os.getenv(ENV_PROMETHEUS_MULTIPROC_DIR, "").strip()
    return value or None


def is_reload_invocation() -> bool:
    """启发式判断：当前是否在 `uvicorn --reload` 模式下运行。

    监控方案 §9 PR-1 要求此场景必须 fail-fast——reloader fork 出的子进程 pid
    会被复用，多进程样本聚合会出现"鬼魂值"。
    """

    if os.getenv(ENV_OBS_ALLOW_RELOAD_MULTIPROC, "").strip() == "1":
        # 高级场景：调用方已经自己处理了 pid 复用，跳过 fail-fast。
        return False
    argv = " ".join(sys.argv).lower()
    return "--reload" in argv


def assert_multiproc_safe() -> None:
    """在 startup 时调用：reload + multiproc 同时启用即直接抛错，避免污染。

    只在 multiproc dir 已配置时检查。单进程模式下 `--reload` 完全合法。
    """

    if not get_multiproc_dir():
        return
    if is_reload_invocation():
        raise RuntimeError(
            "PROMETHEUS_MULTIPROC_DIR is set but the process appears to run in "
            "uvicorn --reload mode. Reloader-forked workers reuse pids and will "
            "pollute multiprocess samples. Either disable reload or unset "
            "PROMETHEUS_MULTIPROC_DIR (set OBS_ALLOW_RELOAD_MULTIPROC=1 to override)."
        )


def lifespan_should_prepare_multiproc_dir() -> bool:
    """lifespan 是否应该兜底清理 multiproc dir。

    监控方案 §9 PR-1 修订：默认 **不** 在 lifespan 里清。
    多 worker 模式下 entrypoint 已经清过一次；如果 lifespan 也清，
    后启动 worker 会把先启动 worker 的样本擦掉。

    单进程模式下没必要清（multiproc 文件压根不会存在）。
    `OBS_LIFESPAN_PREPARE_MULTIPROC_DIR=1` 留给"本地单进程 + 想让 app 自己接管"的场景。
    """

    return os.getenv(ENV_OBS_LIFESPAN_PREPARE_MULTIPROC_DIR, "").strip() == "1"


def prepare_multiproc_dir() -> None:
    """启动前清空 multiproc dir，确保上次崩溃残留不会让计数翻倍。

    Dockerfile entrypoint 也会做一次同样的清理，这里属于代码层兜底，
    本地裸跑（无 entrypoint）也能享受到。
    """

    multiproc_dir = get_multiproc_dir()
    if not multiproc_dir:
        return
    path = Path(multiproc_dir)
    if path.exists():
        for entry in path.iterdir():
            if entry.is_file() or entry.is_symlink():
                try:
                    entry.unlink()
                except OSError as exc:
                    logger.warning(
                        {
                            "event": "observability_emit_failed",
                            "stage": "multiproc_cleanup",
                            "path": str(entry),
                            "error": str(exc),
                        }
                    )
            elif entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def mark_process_dead_safe(pid: int) -> None:
    """优雅退出钩子：回收某 worker 的样本文件。"""

    if not get_multiproc_dir():
        return
    try:
        multiprocess.mark_process_dead(pid)
    except Exception as exc:  # noqa: BLE001 — fire-and-forget
        logger.warning(
            {
                "event": "observability_emit_failed",
                "stage": "mark_process_dead",
                "pid": pid,
                "error": str(exc),
            }
        )


def _pid_alive(pid: int) -> bool:
    """无副作用地探测 pid 是否仍存活（向 pid 发 0 信号）。"""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # 我们能"看见"但没权限给它发信号 → 进程仍存活。
        return True
    except OSError:
        return False
    return True


def scan_stale_pids(reported: set[int] | None = None) -> int:
    """巡检 multiproc dir，清理 pid 已不存在的 live gauge db 文件。

    实现要点（修订）：

    - **只扫描 `gauge_livesum_<pid>.db` / `gauge_liveall_<pid>.db`**。这是
      `prometheus_client.multiprocess.mark_process_dead(pid)` 唯一会处理的两类
      文件；counter / histogram / 其它 gauge 模式必须保留，否则会丢累计样本。
    - **`reported` 集合去重**。即使 mark_process_dead 由于 race 没把文件删干净，
      下一轮扫描也不会再次报警。调用方可传入跨调用复用的集合。

    返回本次新清理的 pid 数；同时按数量上报 `*_multiproc_dir_anomaly_total`。
    """

    multiproc_dir = get_multiproc_dir()
    if not multiproc_dir:
        return 0
    path = Path(multiproc_dir)
    if not path.is_dir():
        return 0

    seen: set[int] = set()
    cleaned: set[int] = set()
    for entry in path.iterdir():
        if not entry.is_file():
            continue
        match = _LIVE_GAUGE_FILE.match(entry.name)
        if not match:
            # counter/histogram 等文件即使 pid 死了也要留——不报、不清。
            continue
        pid = int(match.group(2))
        if pid in seen:
            continue
        seen.add(pid)
        if reported is not None and pid in reported:
            continue
        if _pid_alive(pid):
            continue
        # pid 已死：删掉它的 live gauge 文件
        mark_process_dead_safe(pid)
        cleaned.add(pid)
        if reported is not None:
            reported.add(pid)

    if cleaned:
        emit_counter(
            METRIC_OBSERVABILITY_MULTIPROC_DIR_ANOMALY_TOTAL,
            value=float(len(cleaned)),
        )
        logger.info(
            {
                "event": "multiproc_dir_stale_pids_cleaned",
                "pids": sorted(cleaned),
            }
        )
    return len(cleaned)


class StalePidScanner:
    """后台周期任务：周期性 scan_stale_pids。

    生命周期跟随 FastAPI app：startup 启动，shutdown 停止。
    单线程实现，避免引入额外异步依赖；scan 自身吞掉异常，不影响主进程。
    """

    def __init__(self, interval_seconds: float = DEFAULT_SCAN_INTERVAL_SECONDS) -> None:
        self._interval = max(1.0, interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # 跨周期复用的"已报告过的死 pid"集合，防止 mark_process_dead 失败时重复告警。
        self._reported: set[int] = set()

    def start(self) -> None:
        if not get_multiproc_dir():
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        thread = threading.Thread(
            target=self._run, name="multiproc-stale-pid-scanner", daemon=True
        )
        thread.start()
        self._thread = thread

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        # 启动后先等一个周期再扫描，避免和 prepare_multiproc_dir 抢窗口。
        while not self._stop.wait(self._interval):
            try:
                scan_stale_pids(reported=self._reported)
            except Exception as exc:  # noqa: BLE001 — 守护线程必须吞错
                logger.warning(
                    {
                        "event": "observability_emit_failed",
                        "stage": "stale_pid_scanner",
                        "error": str(exc),
                    }
                )
                # 短暂 sleep 让外部条件恢复后再继续
                time.sleep(min(self._interval, 5.0))


__all__ = [
    "ENV_OBS_ALLOW_RELOAD_MULTIPROC",
    "ENV_OBS_LIFESPAN_PREPARE_MULTIPROC_DIR",
    "ENV_PROMETHEUS_MULTIPROC_DIR",
    "StalePidScanner",
    "assert_multiproc_safe",
    "get_multiproc_dir",
    "is_reload_invocation",
    "lifespan_should_prepare_multiproc_dir",
    "mark_process_dead_safe",
    "prepare_multiproc_dir",
    "scan_stale_pids",
]

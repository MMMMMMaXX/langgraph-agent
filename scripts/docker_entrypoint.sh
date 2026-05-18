#!/bin/sh
# 容器入口脚本（PR-1）。
#
# 监控方案 §9 PR-1 要求：
# - 启动前必须清空 PROMETHEUS_MULTIPROC_DIR，避免上次崩溃残留 *.db 让计数翻倍；
# - 优雅退出由 prometheus_client.multiprocess.mark_process_dead 处理（应用层做）；
# - SIGKILL / OOM 残留由代码内 stale pid 巡检兜底（应用层做）。
#
# 该脚本只负责"进程整体启动前的目录清理"这条兜底主路径。

set -e

if [ -n "${PROMETHEUS_MULTIPROC_DIR:-}" ]; then
    rm -rf "${PROMETHEUS_MULTIPROC_DIR:?}"/*
    mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
fi

exec "$@"

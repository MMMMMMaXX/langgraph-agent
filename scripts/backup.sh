#!/usr/bin/env bash
# 备份项目有状态数据：
#   1. 会话历史（SQLite / JSONL）
#   2. Chroma 向量库目录
#
# 用法：
#   scripts/backup.sh                  # 默认落到 ./backups/<timestamp>/，保留 14 天
#   BACKUP_DIR=/mnt/nas scripts/backup.sh
#   BACKUP_RETENTION_DAYS=30 scripts/backup.sh
#
# cron 示例（每天 03:10 跑一次）：
#   10 3 * * * cd /path/to/langgraph-agent && ./scripts/backup.sh >> backups/backup.log 2>&1

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_ROOT="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-14}"

TARGET="$BACKUP_ROOT/$STAMP"
mkdir -p "$TARGET"

echo "[backup] target=$TARGET"

# ---- 1) 会话历史 ----
SQLITE_DB="data/conversation_history.sqlite3"
JSONL_FILE="data/conversation_history.jsonl"
CHECKPOINT_DB="data/langgraph_checkpoints.sqlite3"

backup_sqlite() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$src" ]]; then
    return
  fi
  # .backup 是 SQLite 在线备份命令，运行中也能安全快照
  if command -v sqlite3 >/dev/null 2>&1; then
    sqlite3 "$src" ".backup '$dst'"
    echo "[backup]   sqlite -> $dst"
  else
    echo "[backup]   sqlite3 CLI 不存在，回退为直接 cp: $src"
    cp "$src" "$dst"
  fi
}

backup_sqlite "$SQLITE_DB"     "$TARGET/conversation_history.sqlite3"
backup_sqlite "$CHECKPOINT_DB" "$TARGET/langgraph_checkpoints.sqlite3"

if [[ -f "$JSONL_FILE" ]]; then
  cp "$JSONL_FILE" "$TARGET/conversation_history.jsonl"
  echo "[backup]   jsonl  -> $TARGET/conversation_history.jsonl"
fi

# ---- 2) Chroma 向量库 ----
CHROMA_DIR="data/chroma"
if [[ -d "$CHROMA_DIR" ]]; then
  tar -czf "$TARGET/chroma.tar.gz" -C data chroma
  echo "[backup]   chroma -> $TARGET/chroma.tar.gz"
fi

# ---- 3) 记录元数据 ----
{
  echo "timestamp=$STAMP"
  echo "git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "host=$(hostname)"
} > "$TARGET/MANIFEST"

# ---- 4) 清理过期备份 ----
if [[ "$RETENTION_DAYS" -gt 0 ]]; then
  find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d -mtime "+$RETENTION_DAYS" -print -exec rm -rf {} +
fi

echo "[backup] done: $TARGET"

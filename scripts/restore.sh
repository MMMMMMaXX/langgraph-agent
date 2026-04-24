#!/usr/bin/env bash
# 从 scripts/backup.sh 产生的快照中还原数据。
#
# 用法：
#   scripts/restore.sh 20260423_031005
#   BACKUP_DIR=/mnt/nas scripts/restore.sh 20260423_031005
#
# 注意：还原会覆盖 data/ 下的同名文件，建议先停掉服务：
#   docker compose down           # 或者 Ctrl-C 掉本地 uvicorn

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <backup_timestamp>" >&2
  exit 1
fi

STAMP="$1"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

BACKUP_ROOT="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
SRC="$BACKUP_ROOT/$STAMP"

if [[ ! -d "$SRC" ]]; then
  echo "[restore] not found: $SRC" >&2
  exit 1
fi

echo "[restore] from $SRC"
mkdir -p data

# 兜底确认，避免误操作
read -r -p "will overwrite files in data/. continue? [y/N] " reply
if [[ "$reply" != "y" && "$reply" != "Y" ]]; then
  echo "[restore] aborted"
  exit 0
fi

restore_file() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
    echo "[restore]   $dst"
  fi
}

restore_file "$SRC/conversation_history.sqlite3" "data/conversation_history.sqlite3"
restore_file "$SRC/conversation_history.jsonl"   "data/conversation_history.jsonl"
restore_file "$SRC/langgraph_checkpoints.sqlite3" "data/langgraph_checkpoints.sqlite3"

if [[ -f "$SRC/chroma.tar.gz" ]]; then
  rm -rf data/chroma
  tar -xzf "$SRC/chroma.tar.gz" -C data
  echo "[restore]   data/chroma/"
fi

echo "[restore] done. 如运行在 Docker 中，别忘了 docker compose up -d 重新起服。"

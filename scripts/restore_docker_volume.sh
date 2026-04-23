#!/usr/bin/env bash
set -euo pipefail

VOLUME_NAME="${VOLUME_NAME:-langgraph_data}"
BACKUP_PATH="${BACKUP_PATH:-}"
CONFIRM_RESTORE="${CONFIRM_RESTORE:-}"

if [[ -z "$BACKUP_PATH" ]]; then
  echo "missing BACKUP_PATH, example:" >&2
  echo "  BACKUP_PATH=backups/langgraph_data-20260423-120000.tar.gz CONFIRM_RESTORE=yes $0" >&2
  exit 2
fi

if [[ ! -f "$BACKUP_PATH" ]]; then
  echo "backup file not found: ${BACKUP_PATH}" >&2
  exit 2
fi

if [[ "$CONFIRM_RESTORE" != "yes" ]]; then
  echo "restore is destructive and will replace data in volume: ${VOLUME_NAME}" >&2
  echo "rerun with CONFIRM_RESTORE=yes to continue." >&2
  exit 2
fi

backup_dir="$(cd "$(dirname "$BACKUP_PATH")" && pwd)"
backup_file="$(basename "$BACKUP_PATH")"

echo "restoring docker volume: ${VOLUME_NAME}"
echo "source: ${BACKUP_PATH}"

docker run --rm \
  -v "${VOLUME_NAME}:/volume" \
  -v "${backup_dir}:/backup:ro" \
  alpine:3.20 \
  sh -c "find /volume -mindepth 1 -maxdepth 1 -exec rm -rf {} + && cd /volume && tar -xzf /backup/${backup_file}"

echo "restore completed: ${VOLUME_NAME}"

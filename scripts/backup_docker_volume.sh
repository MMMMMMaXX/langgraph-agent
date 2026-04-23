#!/usr/bin/env bash
set -euo pipefail

VOLUME_NAME="${VOLUME_NAME:-langgraph_data}"
BACKUP_DIR="${BACKUP_DIR:-backups}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_FILE="${BACKUP_FILE:-${VOLUME_NAME}-${TIMESTAMP}.tar.gz}"

mkdir -p "$BACKUP_DIR"

echo "backing up docker volume: ${VOLUME_NAME}"
echo "output: ${BACKUP_DIR}/${BACKUP_FILE}"

docker run --rm \
  -v "${VOLUME_NAME}:/volume:ro" \
  -v "$(pwd)/${BACKUP_DIR}:/backup" \
  alpine:3.20 \
  sh -c "cd /volume && tar -czf /backup/${BACKUP_FILE} ."

echo "backup completed: ${BACKUP_DIR}/${BACKUP_FILE}"

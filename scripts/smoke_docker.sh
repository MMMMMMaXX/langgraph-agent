#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
SESSION_ID="${SESSION_ID:-docker-smoke-$(date +%s)}"
FIRST_MESSAGE="${FIRST_MESSAGE:-你好}"
SUMMARY_MESSAGE="${SUMMARY_MESSAGE:-总结一下问题}"

post_chat() {
  local message="$1"
  python3 - "$BASE_URL" "$SESSION_ID" "$message" <<'PY'
import json
import sys
import urllib.request

base_url, session_id, message = sys.argv[1:4]
payload = json.dumps(
    {"session_id": session_id, "message": message, "debug": True},
    ensure_ascii=False,
).encode("utf-8")
request = urllib.request.Request(
    f"{base_url}/chat",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=60) as response:
    print(response.read().decode("utf-8"))
PY
}

extract_answer() {
  python3 -c 'import json,sys; print(json.load(sys.stdin).get("answer", ""))'
}

echo "[1/5] health check: ${BASE_URL}/health"
curl -fsS "${BASE_URL}/health"
echo

echo "[2/5] first chat turn: ${FIRST_MESSAGE}"
first_response="$(post_chat "$FIRST_MESSAGE")"
echo "$first_response"

echo "[3/5] docker compose restart"
docker compose -f "$COMPOSE_FILE" restart

echo "[4/5] wait for health after restart"
for _ in $(seq 1 30); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    echo "healthy"
    break
  fi
  sleep 1
done
curl -fsS "${BASE_URL}/health" >/dev/null

echo "[5/5] summary turn: ${SUMMARY_MESSAGE}"
summary_response="$(post_chat "$SUMMARY_MESSAGE")"
echo "$summary_response"
summary_answer="$(printf '%s' "$summary_response" | extract_answer)"

if [[ "$summary_answer" != *"$FIRST_MESSAGE"* ]]; then
  echo "smoke failed: summary answer does not contain previous message: ${FIRST_MESSAGE}" >&2
  exit 1
fi

echo "smoke passed: checkpoint/history survived docker restart for session ${SESSION_ID}"

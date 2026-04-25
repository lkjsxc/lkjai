#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="runs/kimi_corpus"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
PID_FILE="$RUN_DIR/kimi_corpus.pid"
RUN_JSON="$RUN_DIR/run.json"
STOP_FILE="$RUN_DIR/STOP"

mkdir -p "$RUN_DIR"
date -u +%Y-%m-%dT%H:%M:%SZ > "$STOP_FILE"

if [[ ! -f "$PID_FILE" ]]; then
  echo "no PID file found at $PID_FILE"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if [[ -z "$PID" ]]; then
  echo "empty PID file"
  rm -f "$PID_FILE"
  exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
  kill -TERM "$PID"
  echo "sent SIGTERM to $PID"
else
  echo "process $PID is not active"
fi

python3 - "$RUN_JSON" <<'PY' || true
import json, sys
path = sys.argv[1]
try:
    data = json.loads(open(path, encoding="utf-8").read())
except Exception:
    data = {}
data["status"] = "stopped"
open(path, "w", encoding="utf-8").write(json.dumps(data, indent=2) + "\n")
PY

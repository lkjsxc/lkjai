#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/corpus/kimi_500m.yaml"
TARGET="500000000"
OUTPUT_DIR=""
MODE="mixed"
PARALLELISM="2"
FORCE=0
DRY_RUN=0
RUN_DIR="runs/kimi_corpus"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --target-tokens) TARGET="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --parallelism) PARALLELISM="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --debug) CONFIG="configs/corpus/kimi_debug.yaml"; TARGET="50000"; shift ;;
    --smoke) CONFIG="configs/corpus/kimi_debug.yaml"; TARGET="1000000"; shift ;;
    --full) CONFIG="configs/corpus/kimi_500m.yaml"; TARGET="500000000"; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PID_FILE="$RUN_DIR/kimi_corpus.pid"
LOG_FILE="$RUN_DIR/background.log"
RUN_JSON="$RUN_DIR/run.json"
STOP_FILE="$RUN_DIR/STOP"

CMD=(python3 tools/kimi-corpus/generate_kimi_corpus.py --config "$CONFIG" --target-tokens "$TARGET" --mode "$MODE" --parallelism "$PARALLELISM" --run-dir "$RUN_DIR" --resume)
if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output-dir "$OUTPUT_DIR")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'dry-run command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$RUN_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null && [[ "$FORCE" != "1" ]]; then
    echo "kimi corpus generation already appears active with PID $OLD_PID" >&2
    exit 1
  fi
fi

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || true)"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cat > "$RUN_JSON" <<JSON
{
  "status": "dry-run",
  "started_at": "$STARTED_AT",
  "git_commit": "$GIT_SHA",
  "config": "$CONFIG",
  "target_tokens": $TARGET,
  "mode": "$MODE",
  "parallelism": $PARALLELISM,
  "log_file": "$LOG_FILE",
  "command": "${CMD[*]}"
}
JSON

rm -f "$STOP_FILE"
if command -v setsid >/dev/null 2>&1; then
  setsid "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
else
  nohup "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
fi
PID="$!"
echo "$PID" > "$PID_FILE"
python3 - "$RUN_JSON" "$PID" <<'PY'
import json, sys
path, pid = sys.argv[1], sys.argv[2]
data = json.loads(open(path, encoding="utf-8").read())
data["status"] = "running"
data["pid"] = int(pid)
open(path, "w", encoding="utf-8").write(json.dumps(data, indent=2) + "\n")
PY
echo "started kimi corpus generation pid=$PID log=$LOG_FILE"

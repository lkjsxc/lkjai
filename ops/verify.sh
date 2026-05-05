#!/bin/sh
set -eu

LOG_DIR="${VERIFY_LOG_DIR:-/tmp/lkjai-verify-logs}"
TAIL_LINES="${VERIFY_TAIL_LINES:-120}"
GIT_SHA="$(
  git -c safe.directory=/workspace -C /workspace rev-parse --short HEAD \
    2>/dev/null || true
)"
mkdir -p "$LOG_DIR"

run_step() {
  label="$1"
  shift
  log="$LOG_DIR/$(printf '%s' "$label" | tr ' /' '__').log"
  echo "== $label =="
  if "$@" >"$log" 2>&1; then
    lines="$(wc -l < "$log" | tr -d ' ')"
    echo "pass: $label ($lines log lines, full log: $log)"
    return 0
  else
    status="$?"
  fi
  echo "fail: $label (exit $status, full log: $log)"
  echo "-- tail: $log --"
  tail -n "$TAIL_LINES" "$log" || true
  exit "$status"
}

run_step "native configure" cmake -S native -B /tmp/lkjai-native-build -G Ninja \
  -DLKJAI_GIT_COMMIT_OVERRIDE="${GIT_SHA:-unknown}"
run_step "native build" cmake --build /tmp/lkjai-native-build --parallel
run_step "native tests" ctest --test-dir /tmp/lkjai-native-build --output-on-failure
CHECK=/tmp/lkjai-native-build/lkjai-native-repo-check
run_step "docs topology" "$CHECK" docs-topology --repo /workspace
run_step "docs links" "$CHECK" docs-links --repo /workspace
run_step "corpus actions" "$CHECK" corpus-actions -- \
  /workspace/corpus/generated/kimi-sft-60m-v2/train/train-000001.jsonl \
  /workspace/corpus/generated/kimi-sft-60m-v2/val/val-000001.jsonl \
  /workspace/corpus/generated/kimi-sft-60m-v2/holdout/holdout-000001.jsonl
run_step "line limits" "$CHECK" line-limits --repo /workspace
run_step "forbidden js runtime check" "$CHECK" no-node --repo /workspace
run_step "native-only workflow check" "$CHECK" native-only --repo /workspace

echo "== gates passed; logs: $LOG_DIR =="

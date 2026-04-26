#!/bin/sh
set -eu

LOG_DIR="${VERIFY_LOG_DIR:-/tmp/lkjai-verify-logs}"
TAIL_LINES="${VERIFY_TAIL_LINES:-120}"
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

run_step "rust fmt" cargo fmt --all -- --check
run_step "rust tests" cargo test --workspace
run_step "python tests" python3 -m pytest -o cache_dir=/tmp/pytest-cache -m "not slow" training/tests
run_step "docs topology" cargo run -p lkjai --bin lkjai -- docs validate-topology
run_step "docs links" cargo run -p lkjai --bin lkjai -- docs validate-links
run_step "line limits" cargo run -p lkjai --bin lkjai -- quality check-lines
run_step "forbidden js runtime check" cargo run -p lkjai --bin lkjai -- quality no-node

echo "== gates passed; logs: $LOG_DIR =="

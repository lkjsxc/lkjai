#!/usr/bin/env bash
set -euo pipefail

APP_URL="${APP_URL:-http://app:8080}"
export APP_URL

wait_for_healthz() {
  local max_attempts=30
  local attempt=1
  while (( attempt <= max_attempts )); do
    if curl -fsS "${APP_URL}/healthz" >/dev/null; then
      return 0
    fi
    sleep 1
    ((attempt++))
  done
  echo "FAIL: health check did not succeed at ${APP_URL}/healthz after ${max_attempts} attempts"
  return 1
}

./scripts/check_docs_topology.sh
./scripts/check_line_limits.sh

zig fmt --check $(find src -type f -name '*.zig' | sort)
zig build
zig build test
zig build artifact-size-check

wait_for_healthz
./scripts/verify_api_integration.sh

echo "verification passed"

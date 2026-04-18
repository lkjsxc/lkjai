#!/usr/bin/env bash
set -euo pipefail

./scripts/check_docs_topology.sh
./scripts/check_line_limits.sh

zig fmt --check $(find src -type f -name '*.zig' | sort)
zig build
zig build test
zig build artifact-size-check

curl -fsS http://app:8080/healthz >/dev/null

echo "verification passed"

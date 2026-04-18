#!/usr/bin/env bash
set -euo pipefail

fail=0

while IFS= read -r file; do
  lines=$(wc -l < "$file" | tr -d ' ')
  if [[ "$lines" -gt 300 ]]; then
    echo "docs line limit exceeded ($lines): $file"
    fail=1
  fi
done < <(find docs -type f -name '*.md' | sort)

while IFS= read -r file; do
  lines=$(wc -l < "$file" | tr -d ' ')
  if [[ "$lines" -gt 200 ]]; then
    echo "source line limit exceeded ($lines): $file"
    fail=1
  fi
done < <(find src -type f -name '*.zig' | sort)

exit "$fail"


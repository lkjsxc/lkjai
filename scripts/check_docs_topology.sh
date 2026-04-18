#!/usr/bin/env bash
set -euo pipefail

fail=0
while IFS= read -r dir; do
  if [[ ! -f "$dir/README.md" ]]; then
    echo "missing README.md: $dir"
    fail=1
    continue
  fi
  child_count=$(find "$dir" -mindepth 1 -maxdepth 1 ! -name 'README.md' ! -name '.*' | wc -l | tr -d ' ')
  if [[ "$child_count" -lt 2 ]]; then
    echo "directory must have multiple children: $dir"
    fail=1
  fi
done < <(find docs -type d | sort)

exit "$fail"


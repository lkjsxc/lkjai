#!/bin/sh
set -eu

check_bin="$1"
root="/dev/shm/lkjai-native-only-source-scan"
log="$root/check.log"

rm -rf "$root"
mkdir -p "$root/native/src"
{
  printf 'const char* workflow = "'
  printf 'python3 '
  printf 'tools/bad.py";\n'
} > "$root/native/src/bad.cpp"

if "$check_bin" native-only --repo "$root" > "$log" 2>&1; then
  echo "bad native-only source scan unexpectedly passed" >&2
  exit 1
fi

grep -q "bad.cpp contains forbidden workflow" "$log"

#!/usr/bin/env bash
set -euo pipefail

limit=512
size_file="artifacts/deploy-model.size-mib"

if [[ ! -f "$size_file" ]]; then
  echo "missing artifact metadata: $size_file"
  exit 1
fi

size=$(tr -d ' \n\r\t' < "$size_file")
if [[ ! "$size" =~ ^[0-9]+$ ]]; then
  echo "invalid artifact size metadata: $size"
  exit 1
fi

if (( size > limit )); then
  echo "deploy artifact too large: ${size}MiB (limit ${limit}MiB)"
  exit 1
fi

echo "artifact size check passed: ${size}MiB <= ${limit}MiB"


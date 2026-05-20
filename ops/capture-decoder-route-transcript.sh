#!/usr/bin/env bash
set -euo pipefail

url="${MODEL_API_URL:-http://127.0.0.1:8081/v1/chat/completions}"
model="${MODEL_NAME:-decoder-40m-3070}"
report="${TRAIN_REPORT:-data/train/runs/train-report.json}"
manifest="${ARTIFACT_MANIFEST:-data/models/${model}/manifest.json}"
out="${ROUTE_TRANSCRIPT:-data/train/runs/${model}-route-transcript.json}"

mkdir -p "$(dirname "$out")"
request="$(mktemp)"
response="$(mktemp)"
trap 'rm -f "$request" "$response"' EXIT

cat > "$request" <<JSON
{"model":"${model}","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"temperature":0}
JSON

status="$(curl -sS -o "$response" -w '%{http_code}' \
  -H 'content-type: application/json' --data-binary "@${request}" "$url")"

field() {
  sed -n "s/.*\"$1\":\"\\([^\"]*\\)\".*/\\1/p" "$response" | head -1
}

number() {
  sed -n "s/.*\"$1\":\\([0-9][0-9]*\\).*/\\1/p" "$response" | head -1
}

choices=false
grep -q '"choices"' "$response" && choices=true
created_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
train_digest="$(sha256sum "$report" | awk '{print $1}')"
artifact_digest="$(sha256sum "$manifest" | awk '{print $1}')"
decode_backend="$(field lkjai_decode_backend)"
kv_backend="$(field lkjai_kv_cache_backend)"
prefill="$(number lkjai_kv_prefill_allocated_bytes)"
steady="$(number lkjai_kv_steady_state_token_allocations)"

cat > "$out" <<JSON
{"route":"/v1/chat/completions","request":$(cat "$request"),"response_status":${status},"choices_present":${choices},"decode_backend":"${decode_backend}","kv_cache_backend":"${kv_backend}","kv_cache_prefill_allocated_bytes":${prefill:-0},"kv_cache_steady_state_token_allocations":${steady:-0},"train_report_digest":"${train_digest}","artifact_manifest_digest":"${artifact_digest}","created_at":"${created_at}"}
JSON

cat "$out"

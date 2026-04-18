#!/usr/bin/env bash
set -euo pipefail

APP_URL="${APP_URL:-http://app:8080}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"
STATUS_MARKER="__HTTP_STATUS__"

if [[ -z "$ADMIN_TOKEN" ]]; then
  echo "FAIL: ADMIN_TOKEN is required for API integration verification"
  exit 1
fi

REQUEST_CODE=""
REQUEST_BODY=""

run_request() {
  local method="$1"
  local path="$2"
  local payload="${3:-}"
  local use_token="$4"
  local url="${APP_URL}${path}"
  local response

  local args=(
    --silent
    --show-error
    --max-time 15
    --connect-timeout 5
    --request "$method"
    --url "$url"
    --write-out "${STATUS_MARKER}%{http_code}"
  )

  if [[ "$use_token" == "yes" ]]; then
    args+=(--header "x-admin-token: ${ADMIN_TOKEN}")
  fi
  if [[ -n "$payload" ]]; then
    args+=(--header "content-type: application/json" --data "$payload")
  fi

  response="$(curl "${args[@]}")"
  REQUEST_CODE="${response##*${STATUS_MARKER}}"
  REQUEST_BODY="${response%${STATUS_MARKER}*}"
}

fail_with_response() {
  local message="$1"
  echo "FAIL: ${message}"
  echo "  status=${REQUEST_CODE}"
  echo "  body=${REQUEST_BODY}"
  exit 1
}

expect_code() {
  local expected="$1"
  local message="$2"
  if [[ "$REQUEST_CODE" != "$expected" ]]; then
    fail_with_response "${message} (expected ${expected})"
  fi
}

expect_contains() {
  local needle="$1"
  local message="$2"
  if ! grep -Fq -- "$needle" <<<"$REQUEST_BODY"; then
    fail_with_response "${message} (missing: ${needle})"
  fi
}

expect_not_contains() {
  local needle="$1"
  local message="$2"
  if grep -Fq -- "$needle" <<<"$REQUEST_BODY"; then
    fail_with_response "${message} (unexpected: ${needle})"
  fi
}

pass_step() {
  echo "PASS: $1"
}

record_id="verify-integration-record"
record_title="verify-integration-title"
record_body="deterministic integration body"

run_request "GET" "/api/records/list?q=verify" "" "no"
expect_code "401" "unauthorized list request should be rejected"
expect_contains "\"code\":\"unauthorized\"" "unauthorized list response should include error code"
pass_step "unauthorized GET /api/records/list rejected"

run_request "POST" "/api/chat" "{\"message\":\"unauthorized check\"}" "no"
expect_code "401" "unauthorized chat request should be rejected"
expect_contains "\"code\":\"unauthorized\"" "unauthorized chat response should include error code"
pass_step "unauthorized POST /api/chat rejected"

run_request "POST" "/api/records/upsert" "{\"id\":\"${record_id}\",\"title\":\"${record_title}\",\"body\":\"${record_body}\"}" "no"
expect_code "401" "unauthorized upsert request should be rejected"
expect_contains "\"code\":\"unauthorized\"" "unauthorized upsert response should include error code"
pass_step "unauthorized POST /api/records/upsert rejected"

run_request "POST" "/api/records/delete" "{\"id\":\"${record_id}\"}" "no"
expect_code "401" "unauthorized delete request should be rejected"
expect_contains "\"code\":\"unauthorized\"" "unauthorized delete response should include error code"
pass_step "unauthorized POST /api/records/delete rejected"

run_request "POST" "/api/records/delete" "{\"id\":\"${record_id}\"}" "yes"
if [[ "$REQUEST_CODE" != "200" && "$REQUEST_CODE" != "404" ]]; then
  fail_with_response "cleanup delete should return 200 or 404"
fi
pass_step "record cleanup executed"

run_request "POST" "/api/records/upsert" "{\"id\":\"${record_id}\",\"title\":\"${record_title}\",\"body\":\"${record_body}\"}" "yes"
expect_code "200" "record upsert should succeed"
expect_contains "\"status\":\"ok\"" "record upsert response should report ok"
pass_step "record upsert succeeded"

run_request "GET" "/api/records/list?q=${record_title}" "" "yes"
expect_code "200" "record list should succeed"
expect_contains "\"id\":\"${record_id}\"" "record list should include inserted id"
expect_contains "\"title\":\"${record_title}\"" "record list should include inserted title"
pass_step "record list includes inserted record"

run_request "POST" "/api/chat" "{\"message\":\"verify deterministic envelope\"}" "yes"
expect_code "200" "chat request should succeed"
expect_contains "\"status\":\"ok\"" "chat response should include status"
expect_contains "\"reply\":" "chat response should include reply"
expect_contains "\"trace\":" "chat response should include trace"
expect_contains "\"parallel_steps\":" "chat response should include trace.parallel_steps"
expect_contains "\"queue_depth\":" "chat response should include trace.queue_depth"
pass_step "chat JSON envelope verified"

run_request "POST" "/api/records/delete" "{\"id\":\"${record_id}\"}" "yes"
expect_code "200" "record delete should succeed"
expect_contains "\"status\":\"ok\"" "record delete response should report ok"
pass_step "record delete succeeded"

run_request "POST" "/api/records/delete" "{\"id\":\"${record_id}\"}" "yes"
expect_code "404" "second record delete should return not_found"
expect_contains "\"code\":\"not_found\"" "second delete response should include not_found code"
pass_step "deleted record reports not_found on second delete"

run_request "GET" "/api/records/list?q=${record_title}" "" "yes"
expect_code "200" "record list after delete should succeed"
expect_not_contains "\"id\":\"${record_id}\"" "deleted record should not appear in list"
pass_step "deleted record absent from list"

echo "API integration verification passed"

import json
import time

from .api import call_kimi, parse_kimi_rows
from .common import estimated_tokens_for_row
from .generate_ids import next_shard_id
from .prompting import build_repair_messages
from .row import normalize_row
from .validate import validate_row


def process_response(
    config,
    api_key,
    messages,
    response,
    batch_size,
    split,
    family,
    ordinal,
    quarantine,
    state,
):
    result = empty_batch_result(split)
    if response.unauthorized:
        return fail_batch(result, "fail", "unauthorized", response.error)
    if response.access_terminated:
        return fail_batch(result, "fail", "access_terminated", response.error)
    if response.quota_exhausted:
        result["stop_reason"] = "quota_exhausted"
        return result
    if response.status != "pass":
        return fail_batch(result, "fail", "api_error", response.error)
    rows, repair_calls, errors = parsed_rows(
        api_key, config, messages, response.text, batch_size
    )
    result["api_calls"] += repair_calls
    result["errors"].extend(errors)
    if rows is None:
        result["rejected_rows"] += batch_size
        return result
    valid_rows, validation_repair_calls = accept_rows_with_repair(
        rows, config, api_key, messages, split, family, ordinal, state, result
    )
    result["api_calls"] += validation_repair_calls
    if valid_rows:
        write_shard(quarantine, split, valid_rows)
        result["generated_rows"] = len(valid_rows)
        result["token_estimate"] = sum(estimated_tokens_for_row(row) for row in valid_rows)
    return result


def call_with_retries(api_key, config, messages):
    max_retries = max(0, int(config.get("max_retries", 0)))
    calls = 0
    response = None
    for attempt in range(max_retries + 1):
        response = call_kimi(api_key, config, messages)
        calls += max(1, response.attempts or 1)
        if response.status == "pass" or response.quota_exhausted or response.unauthorized:
            break
        if response.retryable and attempt < max_retries:
            time.sleep(min(30, 2 ** attempt))
            continue
        break
    return {"response": response, "api_calls": calls}


def parsed_rows(api_key, config, messages, text, batch_size):
    response_text = text
    errors = []
    calls = 0
    for repair in range(int(config.get("repair_attempts", 0)) + 1):
        try:
            return parse_kimi_rows(response_text), calls, errors
        except (json.JSONDecodeError, ValueError) as exc:
            parse_error = exc
            if repair >= int(config.get("repair_attempts", 0)):
                break
            repair_result = call_kimi(api_key, config, build_repair_messages(messages, response_text))
            calls += 1
            if repair_result.status != "pass":
                errors.append(repair_result.error)
                break
            response_text = repair_result.text
    errors.append(f"malformed response: {parse_error}")
    return None, calls, errors


def accept_rows(rows, config, split, family, ordinal, state, result):
    valid_rows = []
    for offset, raw in enumerate(rows):
        row = normalize_row(raw, config, split, family, ordinal, offset)
        local_seen = set(state["ids"])
        row_errors = validate_row(row, split, local_seen)
        if row_errors:
            result["rejected_rows"] += 1
            result["errors"].append("; ".join(row_errors))
            continue
        state["ids"] = local_seen
        valid_rows.append(row)
    return valid_rows


def accept_rows_with_repair(rows, config, api_key, messages, split, family, ordinal, state, result):
    calls = 0
    errors_before = len(result["errors"])
    rejected_before = result["rejected_rows"]
    valid_rows = accept_rows(rows, config, split, family, ordinal, state, result)
    if valid_rows or int(config.get("repair_attempts", 0)) <= 0:
        return valid_rows, calls
    validation_errors = result["errors"][errors_before:]
    for _repair in range(int(config.get("repair_attempts", 0))):
        repair_payload = {
            "rows": rows,
            "validation_errors": validation_errors[:20],
            "repair_kind": "row_validation",
        }
        repair_result = call_kimi(
            api_key,
            config,
            build_repair_messages(messages, json.dumps(repair_payload, ensure_ascii=False)),
        )
        calls += max(1, repair_result.attempts or 1)
        if repair_result.status != "pass":
            result["errors"].append(repair_result.error)
            break
        repaired, parse_calls, parse_errors = parsed_rows(
            api_key, config, messages, repair_result.text, int(config.get("batch_documents", 1))
        )
        calls += parse_calls
        result["errors"].extend(parse_errors)
        if repaired is None:
            continue
        rows = repaired
        result["errors"] = result["errors"][:errors_before]
        result["rejected_rows"] = rejected_before
        valid_rows = accept_rows(rows, config, split, family, ordinal, state, result)
        if valid_rows:
            return valid_rows, calls
        validation_errors = result["errors"][errors_before:]
    return valid_rows, calls


def write_shard(quarantine, split, rows):
    shard = quarantine / split / f"shard-{next_shard_id(quarantine, split):06d}.jsonl"
    tmp = shard.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    tmp.replace(shard)


def empty_batch_result(split):
    return {
        "status": "pass",
        "stop_reason": "",
        "split": split,
        "api_calls": 0,
        "generated_rows": 0,
        "rejected_rows": 0,
        "token_estimate": 0,
        "errors": [],
    }


def fail_batch(result, status, stop_reason, error):
    result["status"] = status
    result["stop_reason"] = stop_reason
    result["errors"].append(error)
    return result

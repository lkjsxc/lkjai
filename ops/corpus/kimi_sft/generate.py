import json
import os
import time
from pathlib import Path

from .common import estimated_tokens_for_row, iter_jsonl, load_api_key, read_config
from .common import write_json
from .constants import CORPUS, DEFAULT_MODEL, DEFAULT_QUARANTINE
from .constants import SPLITS
from .batch import generate_batch, generate_parallel_batches
from .prompting import fixtures, prompt_asset
from .validate import validate_row


def existing_valid_state(root):
    seen = set()
    split_counts = {split: 0 for split in SPLITS}
    token_estimate = 0
    rows = 0
    for split, _path, _number, row, parse_error in iter_jsonl(root):
        if parse_error:
            continue
        row_seen = set(seen)
        if validate_row(row, split, row_seen):
            continue
        seen = row_seen
        rows += 1
        split_counts[split] += 1
        token_estimate += estimated_tokens_for_row(row)
    return {"ids": seen, "rows": rows, "split_rows": split_counts, "token_estimate": token_estimate}


def generate(args):
    config = read_config(args.config)
    apply_pilot_gate(config)
    quarantine = Path(config.get("quarantine_dir") or args.quarantine or DEFAULT_QUARANTINE)
    setup_quarantine(quarantine)
    api_key = load_api_key()
    fixture_rows = fixtures(config)
    prompt_template = prompt_asset(config)
    state = existing_valid_state(quarantine)
    report = generation_report(config, quarantine, state)
    ordinal = state["rows"]
    while should_continue(config, report):
        stop = stop_reason(config, report)
        if stop:
            report["stop_reason"] = stop
            break
        width = dispatch_width(config, report)
        if width > 1:
            result = generate_parallel_batches(
                config, api_key, prompt_template, fixture_rows, quarantine, state, ordinal, width
            )
        else:
            result = generate_batch(config, api_key, prompt_template, fixture_rows, quarantine, state, ordinal)
        report["api_calls"] += result["api_calls"]
        report["generated_rows"] += result["generated_rows"]
        report["rejected_rows"] += result["rejected_rows"]
        report["errors"].extend(result["errors"])
        if result["stop_reason"]:
            report["status"] = result["status"]
            report["stop_reason"] = result["stop_reason"]
            break
        if result.get("batches"):
            for batch in result["batches"]:
                if batch["generated_rows"]:
                    report["split_rows"][batch["split"]] += batch["generated_rows"]
            report["token_estimate"] += result["token_estimate"]
        elif result["generated_rows"]:
            split = result["split"]
            report["split_rows"][split] += result["generated_rows"]
            report["token_estimate"] += result["token_estimate"]
        ordinal += width
        sleep = float(config.get("sleep_between_calls", 0))
        if sleep > 0:
            time.sleep(sleep)
    if not report["stop_reason"]:
        report["stop_reason"] = "target_reached"
    report["finished_unix"] = int(time.time())
    write_json(quarantine / "generation-report.json", report)
    return report


def setup_quarantine(quarantine):
    quarantine.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (quarantine / split).mkdir(parents=True, exist_ok=True)


def generation_report(config, quarantine, state):
    return {
        "schema": "lkjai-kimi-sft-generation",
        "status": "pass",
        "corpus": CORPUS,
        "target_tokens": int(config.get("target_tokens", 0)),
        "full_target_tokens": int(config.get("full_target_tokens", config.get("target_tokens", 0))),
        "pilot_gate_required": bool(config.get("pilot_gate_required", False)),
        "target_multi_turn_ratio": float(config.get("multi_turn_ratio", 0.60)),
        "target_compaction_ratio": float(config.get("compaction_ratio", 0.25)),
        "quarantine": str(quarantine),
        "api_provider": config.get("api_provider", "kimi-api"),
        "api_model": config.get("api_model", DEFAULT_MODEL),
        "token_source": "KIMI_API_KEY_FILE/KIMI_API_KEY",
        "started_unix": int(time.time()),
        "stop_reason": "",
        "existing_rows": state["rows"],
        "generated_rows": 0,
        "rejected_rows": 0,
        "api_calls": 0,
        "split_rows": state["split_rows"],
        "token_estimate": state["token_estimate"],
        "errors": [],
    }


def should_continue(config, report):
    target = int(config.get("target_tokens", 0))
    return target <= 0 or report["token_estimate"] < target


def stop_reason(config, report):
    if Path(str(config.get("stop_file", "runs/kimi_corpus/STOP"))).exists():
        return "stop_file"
    max_calls = int(config.get("max_calls", 0))
    if max_calls and report["api_calls"] >= max_calls:
        return "max_calls"
    return ""


def dispatch_width(config, report):
    width = max(1, int(config.get("parallelism", 1)))
    max_calls = int(config.get("max_calls", 0))
    if max_calls:
        width = min(width, max(0, max_calls - report["api_calls"]))
    return max(1, width)


def apply_pilot_gate(config):
    target = int(config.get("target_tokens", 0))
    pilot = int(config.get("pilot_tokens", 0))
    approved = os.environ.get("KIMI_SFT_FULL_RUN_APPROVED", "").strip() == "1"
    if pilot > 0 and target > pilot and not approved:
        config["full_target_tokens"] = target
        config["target_tokens"] = pilot
        config["pilot_gate_required"] = True

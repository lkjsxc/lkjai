import json
import random
from pathlib import Path

from .constants import DEFAULT_MODEL, TOOLS


def fixtures(config):
    loaded = []
    for file_name in config.get("fixture_files") or []:
        with Path(file_name).open("r", encoding="utf-8") as handle:
            loaded.extend(json.load(handle))
    if not loaded:
        raise SystemExit("no fixture rows loaded")
    return loaded


def prompt_asset(config):
    prompt_dir = Path(config.get("prompt_dir", "tools/kimi-corpus/prompts"))
    asset = prompt_dir / "agent-action-sft.md"
    if asset.is_file():
        return asset.read_text(encoding="utf-8")
    return "Generate valid lkjai agent action SFT rows."


def build_generation_messages(config, prompt_template, fixture_rows, split, family, batch_size, ordinal):
    rng = random.Random(int(config.get("seed", 42)) + ordinal)
    limit = min(len(fixture_rows), batch_size + 2)
    chosen = [fixture_rows[(ordinal + offset) % len(fixture_rows)] for offset in range(limit)]
    rng.shuffle(chosen)
    system = (
        "You generate commercial-safe first-party SFT corpus rows for lkjai. "
        "Return only English JSON. Do not include markdown. Do not reveal secrets. "
        "Do not call or ask Kimi Code CLI to call any tools; generate offline JSON only. "
        "Every assistant message must contain exactly one <action> block with exactly one <tool> tag. "
        "Use only these tools: " + ", ".join(sorted(TOOLS)) + ". "
        "Do not execute mutation tools directly; use agent.request_confirmation with pending_tool instead. "
        "Rows must use schema lkjai-agent-jsonl and provenance kimi-generated. "
        "Rows must be valid lkjai-agent-jsonl objects with messages, tags, and meta."
    )
    constraints = generation_constraints(config, ordinal)
    user = {
        "task": "generate_agent_action_sft_batch",
        "template": prompt_template,
        "batch_size": batch_size,
        "split": split,
        "template_family": family,
        "id_prefix": f"kimi-{split}-{ordinal:09d}",
        "scenario_family_prefix": f"kimi-{family}-{split}-{ordinal:09d}",
        "required_meta": required_meta(config, split, family),
        "constraints": constraints,
        "fixture_context": json.dumps(chosen, ensure_ascii=False),
        "output_contract": {
            "type": "object",
            "key": "rows",
            "rows": "array of JSONL row objects with messages, tags, meta",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def generation_constraints(config, ordinal):
    multi_turn_ratio = float(config.get("multi_turn_ratio", 0.60))
    compaction_ratio = float(config.get("compaction_ratio", 0.25))
    bucket = ordinal % 100
    requires_compaction = bucket < int(compaction_ratio * 100)
    requires_multi_turn = requires_compaction or bucket < int(multi_turn_ratio * 100)
    constraints = {
        "english_only": True,
        "no_markdown": True,
        "no_kimi_tool_use": True,
        "requires_multi_turn": requires_multi_turn,
        "requires_compaction": requires_compaction,
        "multi_turn_contract": (
            "When requires_multi_turn is true, every row must contain at least two user "
            "messages, at least two assistant <action> messages, and at least one tool "
            "message containing a plausible fixture-derived result."
        ),
        "compaction_contract": (
            "When requires_compaction is true, the first message must be a system message "
            "whose content starts with 'Conversation summary: ', meta.compaction must be "
            "true, and tags must include compacted-context."
        ),
    }
    return constraints


def required_meta(config, split, family):
    return {
        "schema": "lkjai-agent-jsonl",
        "split": split,
        "provenance": "kimi-generated",
        "author_type": "external-agent-generated",
        "author_model": str(config.get("api_model", DEFAULT_MODEL)),
        "quality_tier": "high",
        "toolset": "core",
        "language": "en",
        "safety_scope": "workspace-safe",
        "license": "project-local",
        "mode": "sft",
        "prompt_contract": "agent-api",
        "template_family": family,
        "grounding_source": "repo_fixture",
        "contract_validated": True,
        "fixture_executed": True,
    }


def build_repair_messages(original_messages, bad_text):
    repair = {
        "task": "repair_agent_action_sft_batch",
        "reason": "The previous response was not valid JSON matching the requested rows contract.",
        "requirements": [
            "Return only an English JSON object with key rows and no markdown.",
            "Every row must have messages, tags, and meta.",
            "Every assistant message must contain exactly one <action> block and one allowed <tool>.",
            "Mutation operations must use agent.request_confirmation instead of direct mutation tools.",
            "Preserve any requested multi-turn and compaction constraints from the original request.",
        ],
        "allowed_tools": sorted(TOOLS),
        "previous_response_prefix": bad_text[:6000],
    }
    return [
        original_messages[0],
        original_messages[1],
        {"role": "user", "content": json.dumps(repair, ensure_ascii=False)},
    ]

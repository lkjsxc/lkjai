# Schema

## Paths

Historical committed corpus:

```text
corpus/generated/kimi-full-v1/
  README.md
  manifest.json
  train/train-*.jsonl
  val/val-*.jsonl
  holdout/holdout-*.jsonl
```

Runtime staging:

```text
data/kimi_synthetic/
  manifest.jsonl
  sft/{train,val,holdout}/*.jsonl
  quarantine/
```

Pilot committed corpus:

```text
corpus/generated/kimi-sft-pilot-v1/
  README.md
  manifest.jsonl
  validation-report.json
  train/train-*.jsonl
  val/val-*.jsonl
  holdout/holdout-*.jsonl
```

`kimi-full-v1` remains the active training input until the pilot is validated.
Runtime staging is an operator workspace for fresh Kimi API generation before
validation and normalization.

## Active Committed Row

Each JSONL line in committed Kimi SFT corpora uses the XML-action SFT schema.
Active rows must be English-only.

```json
{
  "messages": [
    {"role": "user", "content": "Inspect the project docs."},
    {"role": "assistant", "content": "<action><tool>fs.read</tool><args>{\"path\":\"docs/README.md\"}</args></action>"},
    {"role": "tool", "content": "Docs table of contents..."},
    {"role": "assistant", "content": "<action><tool>agent.finish</tool><content>Summary...</content></action>"}
  ],
  "tags": ["kimi_corpus", "agentic"],
  "meta": {
    "id": "kimi-full-v1-train-000001",
    "split": "train",
    "source": "lkjai-docs",
    "license": "project-local",
    "template_family": "read_only_retrieval",
    "scenario_family_id": "docs-readme-summary",
    "intent": "workspace.read_doc",
    "tool_sequence": ["fs.read", "agent.finish"],
    "confirmation_required": false,
    "grounding_source": "docs/README.md",
    "gold_stop_reason": "finish"
  }
}
```

Every assistant message must contain exactly one XML `<action>`. In SFT rows,
the final assistant action must be `agent.finish`.

## SFT Row

Staging SFT rows use the existing chat row format:

```json
{
  "messages": [
    {"role": "user", "content": "Summarize this note."},
    {"role": "assistant", "content": "<action><tool>agent.finish</tool><content>...</content></action>"}
  ],
  "tags": ["kimi_synthetic", "language:en"],
  "meta": {
    "id": "kimi-sft-000001",
    "split": "train",
    "provenance": "kimi-generated",
    "author_type": "external-agent-generated",
    "author_model": "kimi-code",
    "quality_tier": "high",
    "domain": "summarization",
    "skill": "instruction-following",
    "toolset": "none",
    "language": "en",
    "safety_scope": "workspace-safe",
    "license": "project-local",
    "source_ref": "kimi_synthetic:v2",
    "mode": "sft",
    "prompt_version": "api-v1",
    "template_family": "direct_finish",
    "scenario_family_id": "direct-greeting",
    "intent": "direct_answer.greeting",
    "tool_sequence": ["agent.finish"],
    "confirmation_required": false,
    "grounding_source": "synthetic",
    "gold_stop_reason": "finish"
  }
}
```

`assistant_masked_sft` masks non-assistant text and trains only on assistant
content. Preference comparisons belong in separate preference-pair artifacts,
not in active SFT rows.

## Manifest Row

Each completed or failed shard appends one manifest row with shard id, mode,
split, path, document counts, API-estimated token counts, validation status,
model, redacted key fingerprint, retry count, duration, and failure reason when
present.

## Split Policy

Split by `scenario_family_id` or normalized hash cluster. Do not split siblings
from the same scenario family across train, val, and holdout.

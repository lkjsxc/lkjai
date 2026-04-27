# Schema

## Paths

Committed corpus:

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
  pretrain/{train,val,holdout}/*.jsonl
  sft/{train,val,holdout}/*.jsonl
  quarantine/
```

The committed corpus is the active training input. Runtime staging is an
operator workspace for fresh Kimi CLI generation before validation and
normalization.

## Active Committed Row

Each JSONL line in `corpus/generated/kimi-full-v1/{train,val,holdout}` uses the
pretraining or XML-action SFT schema. Active rows must be English-only.

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
    "license": "project-local"
  }
}
```

Every assistant message must contain exactly one XML `<action>`. In SFT rows,
the final assistant action must be `agent.finish`.

## Pretraining Row

Staging pretraining rows are standalone documents:

```json
{
  "id": "kimi-pretrain-000001",
  "mode": "pretrain",
  "language": "en",
  "domain": "science",
  "difficulty": "introductory",
  "title": "Photosynthesis Overview",
  "text": "Standalone original document text...",
  "metadata": {
    "source": "kimi_synthetic",
    "mode": "pretrain",
    "generated_at": "2026-04-25T00:00:00Z",
    "prompt_version": "v2",
    "estimated_tokens": 180
  }
}
```

`text` is used directly by `causal_lm_full`; every next token contributes to
loss.

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
    "prompt_version": "v2"
  }
}
```

`assistant_masked_sft` masks non-assistant text and trains only on assistant
content. Preference comparisons belong in separate preference-pair artifacts,
not in active SFT rows.

## Manifest Row

Each completed or failed shard appends one manifest row with shard id, mode,
split, path, document counts, token counts, validation status, Kimi command
variant, retry count, duration, and failure reason when present.

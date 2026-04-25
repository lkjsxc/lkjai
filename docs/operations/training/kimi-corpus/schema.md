# Schema

## Paths

Committed corpus:

```text
training/corpus/kimi-synthetic-v1/
  README.md
  manifest.jsonl
  validation-report.json
  pretrain/train/*.jsonl
  pretrain/val/*.jsonl
  pretrain/holdout/*.jsonl
  sft/train/*.jsonl
  sft/val/*.jsonl
  sft/holdout/*.jsonl
```

Runtime staging:

```text
data/kimi_synthetic/
  manifest.jsonl
  pretrain/{train,val,holdout}/*.jsonl
  sft/{train,val,holdout}/*.jsonl
  quarantine/
```

## Pretraining Row

Each JSONL line is one standalone document:

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

Each JSONL line uses the existing chat row format:

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
content.

## Manifest Row

Each completed or failed shard appends one manifest row with shard id, mode,
split, path, document counts, token counts, validation status, Kimi command
variant, retry count, duration, and failure reason when present.

# Training Corpus

## Goal

Define the canonical training dataset for raw-generation evaluation and
`kjxlkj` API integration.

## Storage Schema

Editable source entries live in JSON array files under
`training/corpus_sources/`; see
[source-corpus.md](source-corpus.md). Generated rows are JSONL in
`lkjai-agent-jsonl-v2`:

```json
{
  "messages": [
    {"role": "user", "content": "<task><request>Summarize ...</request></task>"},
    {"role": "assistant", "content": "{\"kind\":\"final\",\"content\":\"...\"}"}
  ],
  "tags": ["docs_grounding", "language:en"],
  "meta": {
    "id": "docs-lkjai-000001",
    "split": "train",
    "provenance": "project-authored",
    "author_type": "llm-curated",
    "author_model": "gpt-5.4-codex",
    "quality_tier": "high",
    "domain": "lkjai-docs",
    "skill": "grounding",
    "toolset": "none",
    "language": "en",
    "safety_scope": "workspace-safe",
    "license": "project-local",
    "source_ref": "docs/architecture/training/corpus.md"
  }
}
```

## Rules

- `messages`, `tags`, and `meta` are required.
- Roles allowed: `system`, `user`, `assistant`, `tool`.
- `assistant` content must be one valid JSON action.
- `meta.split` is one of `train`, `val`, or `holdout`.
- Public rows must use explicit permissive licenses only.
- The mainline corpus must stay commercial-safe.

## Model-Facing Serialization

- Storage remains JSONL.
- Model-facing text uses paired XML-like sections.
- Prompt construction ends with `<assistant_json>` so the model learns the same
  continuation boundary used during inference.
- Recommended sections: `<policy>`, `<memory>`, `<events>`, `<tool_result>`,
  `<task>`, and `<request>`.
- Assistant outputs stay strict JSON because the runtime validates typed fields
  before execution.

## Dataset Targets

- Total rows: `30000`
- Train rows: `24000`
- Validation rows: `3000`
- Holdout rows: `3000`
- Unique normalized rows: at least `24000`
- Deduplicated tokenizer tokens on the train split: at least `3000000`

## Token Budget

- Parameter count: ~55.8M (scratch-60m preset)
- Chinchilla-optimal tokens: ~1.1T (~20 tokens/parameter)
- Practical train tokens at 30k rows: ~4.5M (~0.08 tokens/parameter)
- Gap is intentional: the default path is from-scratch on limited compute.
- Quality and task diversity matter more than raw token volume for this budget.

## Sources

- Tagged JSON source arrays under `training/corpus_sources/`.
- `lkjai` docs-derived grounding rows.
- `kjxlkj` contract and API rows.
- Synthetic tool and confirmation trajectories tied to the real runtime.
- Safety, privacy, and boundary rows.
- Agentic multi-turn rows with observable plans, tool calls, observations,
  revisions, and finals.
- Carefully selected permissive public rows with explicit provenance.

# Training Corpus

## Goal

Define the canonical training dataset for raw-generation evaluation and
`kjxlkj` API integration.

## Storage Schema

Each row is JSONL in `lkjai-agent-jsonl-v2`:

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

- Total rows: `12000`
- Train rows: `9600`
- Validation rows: `1200`
- Holdout rows: `1200`
- Unique normalized rows: at least `8000`
- Deduplicated tokenizer tokens on the train split: at least `1000000`

## Sources

- `lkjai` docs-derived grounding rows.
- `kjxlkj` contract and API rows.
- Synthetic tool and confirmation trajectories tied to the real runtime.
- Safety, privacy, and boundary rows.
- Carefully selected permissive public rows with explicit provenance.

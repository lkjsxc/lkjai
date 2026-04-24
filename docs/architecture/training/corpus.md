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
    "provenance": "repo-derived",
    "author_type": "repo-derived",
    "author_model": "none",
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
- Default rows must not contain GPT, Kimi, Claude, or other LLM-authored corpus
  text.
- Quarantined source packs must not be consumed by `prepare-corpus`.

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

- Total rows: `6000`
- Train rows: approximately `4800`
- Validation rows: approximately `600`
- Holdout rows: approximately `600`
- Unique normalized rows: at least `5000`
- Deduplicated tokenizer tokens on the train split: at least `500000`

## Token Budget

- Parameter count: ~55.8M (scratch-60m preset)
- Chinchilla-optimal tokens: ~1.1T (~20 tokens/parameter)
- Practical train tokens at 6k rows: ~0.9M (~0.016 tokens/parameter)
- Gap is intentional: the default path is from-scratch on limited compute and
  excludes unreviewed LLM-authored corpus packs.
- Trusted provenance and format alignment matter more than raw token volume for
  this budget.

## Sources

- Canonical docs-derived grounding rows from `docs/**/*.md`.
- Runtime schema and route rows derived from repository source and tests.
- Small deterministic fixtures for tool calls, confirmations, revision, and
  safety.
- Carefully selected permissive public rows with explicit provenance.

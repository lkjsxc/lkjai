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

- Total rows: `60000`
- Train rows: approximately `48000`
- Validation rows: approximately `6000`
- Holdout rows: approximately `6000`
- Unique normalized rows: at least `57000`
- Duplicate rows: at most `1%`
- Deduplicated tokenizer tokens on the train split: at least `5000000`

## Token Budget

- Parameter count: ~55.8M (scratch-60m preset)
- Chinchilla-optimal tokens: ~1.1T (~20 tokens/parameter)
- Practical train tokens at 60k rows: ~9M (~0.16 tokens/parameter)
- Gap remains intentional: the default path is from-scratch on limited compute
  and excludes unreviewed LLM-authored corpus packs.
- Trusted provenance and format alignment matter more than raw token volume for
  this budget.

## Sources

- Docs-derived grounding rows from `docs/**/*.md` (~25%).
- Multi-turn agentic rows derived from docs and source files (~20%).
- Runtime schema and route rows derived from repository source and tests (~15%).
- Test-derived fixtures for tool calls, confirmations, failure diagnosis, and
  revision (~10%).
- Safety, provenance, and artifact hygiene rows from policy docs (~10%).
- Codebase navigation and implementation-planning rows from source files (~8%).
- Preference-style critique/revision rows from deterministic rubrics (~7%).
- Public-import rows are not available locally; quota redistributed to repo-derived
  sources (~5%).

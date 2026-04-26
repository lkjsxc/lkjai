# Training Corpus

## Goal

Define the canonical training dataset for raw-generation evaluation and
`kjxlkj` API integration.

## Storage Schema

Editable source entries live in JSON array files under
`corpus/sources/`; see
[source-corpus.md](source-corpus.md). Generated rows are JSONL in
`lkjai-agent-jsonl-v2`:

```json
{
  "messages": [
    {"role": "user", "content": "<task><request>Summarize ...</request></task>"},
    {"role": "assistant", "content": "<action>\n<tool>agent.finish</tool>\n<content>...</content>\n</action>"}
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
- `assistant` content must be one valid XML action.
- `meta.split` is one of `train`, `val`, or `holdout`.
- Public rows must use explicit permissive licenses only.
- The mainline corpus must stay commercial-safe.
- Default rows must not contain GPT/Codex-authored corpus text.
- Kimi Code may create active corpus rows when metadata declares Kimi provenance.
- Quarantined source packs must not be consumed by `prepare-corpus`.

## Model-Facing Serialization

- Storage remains JSONL.
- Model-facing text uses paired XML-like sections.
- Prompt construction ends with `<assistant_action>` so the model learns the same
  continuation boundary used during inference.
- Recommended sections: `<policy>`, `<memory>`, `<events>`, `<tool_result>`,
  `<task>`, and `<request>`.
- Assistant outputs use one XML action whose child tags become typed tool fields.

## Dataset Targets

- Target tokens: `500000000`
- The full Kimi corpus is committed in chunked JSONL under
  `corpus/generated/kimi-full-v1/`.
- Runtime and training copies may be staged under ignored `data/kimi-corpus/`.
- A small smoke corpus may exist for quick local checks.
- Duplicate rows: at most `1%`
- Deduplicated tokenizer tokens on the train split: at least `450000000`

## Token Budget

- Active parameter count: about 40M (`scratch-40m` preset).
- Later scale target: 60M class (`scratch-60m` preset).
- Chinchilla-optimal tokens for 40M: ~800M (~20 tokens/parameter).
- Practical train tokens at current scale: the committed corpus has about 26M
  tokenizer tokens and must be treated as under-tokened.
- Practical target tokens at 40M scale: 500M (~12.5 tokens/parameter).
- The target remains below the classic 20 tokens/parameter heuristic.
- Quality, tool fidelity, and holdout isolation remain mandatory.

## Kimi Corpus Layout

Committed generated corpus chunks live under:

```
corpus/generated/kimi-full-v1/
  README.md
  manifest.json
  validation-report.json
  train/train-000001.jsonl
  val/val-000001.jsonl
  holdout/holdout-000001.jsonl
```

Each JSONL chunk should contain about `1000` rows. The final chunk in each split
may contain fewer rows.

Generated runtime copies may also live under:

```
data/kimi-corpus/
  train/*.jsonl
  val/*.jsonl
  holdout/*.jsonl
  manifest.json
  validation-report.json
```

- `manifest.json` records schema, row counts, split counts, token counts, and sources.
- `validation-report.json` records total rows, duplicate rate, XML validity
  rate, `agent.finish` termination rate, tool distribution, chunk sizes,
  everyday-chat coverage, and provenance distribution.

## Kimi-Generated Provenance

Active `kimi-generated` rows must use:

- `provenance`: `kimi-generated`
- `author_type`: `external-agent-generated`
- `author_model`: `kimi-code`
- explicit `source_ref` and `license`

These rows are mechanically derived from repository files and runtime schemas,
but the trace assembly and multi-turn structure are Kimi-authored.

## Sources

- Kimi-generated standalone pretraining documents.
- Kimi-generated active XML action traces.
- Everyday conversation and follow-up traces.
- Docs-grounded and source-grounded tasks.
- Runtime tool traces with real observations and explicit `agent.finish`.
- Safety, confirmation, failure recovery, and revision traces.

## Rejection Patterns

- Everyday-chat rows must not finish with `Completed task for ...`.
- Greetings, thanks, and capability questions must not call filesystem tools.
- Repeated failed tool calls must be represented as failures to avoid, not as
  successful target behavior.
- Generic final answers are allowed only in explicit negative preference rows.

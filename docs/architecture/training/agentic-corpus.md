# Agentic Corpus

## Goal

Define multi-turn agentic training rows that teach observable planning,
tool chaining, observation handling, revision, and final answers.

## Message Sequence Schema

A full agentic trajectory follows this order:

1. `user` — task request with `<task>` sections.
2. `assistant` — `{"kind":"plan","content":"concise visible plan"}`.
3. `assistant` — `{"kind":"tool_call","thought":"...","tool":"...","args":{}}`.
4. `tool` — observation result.
5. `assistant` — optional second `tool_call` or `revise` step.
6. `tool` — optional second observation.
7. `assistant` — `{"kind":"final","content":"..."}`.

## Plan Action Rules

- `content` must be a concise visible plan, not a hidden chain-of-thought field.
- Maximum length: 120 characters.
- Must reference at least one concrete tool by name.
- Must describe steps in execution order.
- No prose outside the JSON action.

## Row Kinds

### Planning Rows

- User asks a complex task.
- Assistant emits `plan` before any `tool_call`.
- Tags: `agentic`, `planning`, `multi_turn`.

### Tool Chain Rows

- User asks a task requiring two or more sequential tools.
- Assistant emits `tool_call`, observes, emits second `tool_call`, then `final`.
- Tags: `agentic`, `tool_chain`, `multi_turn`.

### Revision Rows

- User asks a task where the first tool fails or returns insufficient data.
- Assistant revises with a different tool or query.
- Tags: `agentic`, `revision`, `multi_turn`.

## Bucket Targets

- Total agentic rows: `10000`
  - Planning: `3000`
  - Tool chains: `4000`
  - Revision: `3000`

## Quality Rules

- Every trajectory must contain at least one `tool` message.
- Final answers must reference observation content, not generic fallback text.
- Plans must be unique per scenario id after deduplication.
- No hidden reasoning fields; all planning is visible in `plan` actions.

## Sources

- `training/corpus_sources/agentic_plan.json`
- `training/corpus_sources/agentic_tools.json`
- `training/corpus_sources/agentic_revision.json`

## Links

- [corpus.md](corpus.md): overall dataset contract
- [pipeline.md](pipeline.md): training pipeline order
- [reasoning-models.md](../../research/reasoning-models.md): reasoning-budget references

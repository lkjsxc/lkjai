# Kimi Corpus Smoke Report

Date: 2026-04-24

This report is historical. Kimi generation is now optional SFT/tool data, while
the active public pretraining path is Cosmopedia `text`-only English.

## Smoke Parameters

- Data dir: `/tmp/lkjai-kimi-smoke`
- Preset: `quick` (smoke cap at 1M tokens)
- Rows generated: 399
- Tokenizer tokens: 150,825

## Split Counts

| Split | Rows |
|-------|------|
| train | 313 |
| val   | 39 |
| holdout | 47 |

## Quality Gates

| Metric | Result | Threshold |
|--------|--------|-----------|
| XML validity rate | 1.0 | >= 0.95 |
| agent.finish rate | 1.0 | 1.0 (required) |
| Duplicate rate | 0.0 | <= 0.01 |
| Provenance | 100% kimi-generated | required |

## Tool Distribution (assistant actions)

- agent.think: 535
- fs.read: 455
- agent.finish: 449
- fs.list: 135
- shell.exec: 26
- resource.fetch: 4
- resource.search: 4
- agent.request_confirmation: 3
- resource.create_note: 1
- memory.search: 2
- fs.write: 1
- resource.update_resource: 1

## Source/License Mix

- lkjai-docs (project-local): 251
- source-code (project-local): 20
- runtime-schema (project-local): 12
- agentic (project-local): 11
- kjxlkj-api (project-local): 3

## Blockers

- Docker verify must pass through `ops/verify.sh` after layout or corpus
  changes.
- PyTorch/tokenizers not installed on host; token count falls back to character heuristic.
- Full 60M-token SFT generation requires Kimi access and significant disk/memory.

## Residual Risks

- Token count is approximate without a pre-trained tokenizer.
- The current active mix must be regenerated before this smoke report is used
  as evidence for acceptance.
- Full generation time and disk usage for 60M SFT tokens are untested at scale.

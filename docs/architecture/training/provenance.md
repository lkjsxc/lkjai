# Training Data Provenance

## Goal

Keep active training data traceable to project canon, runtime contracts, tests,
or explicit permissive public sources.

## Allowed Active Provenance

- `repo-derived`: generated from `docs/`, source schemas, tests, or manifests.
- `test-derived`: generated from deterministic verification fixtures.
- `runtime-schema-derived`: generated from typed tool, API, or action schemas.
- `human-seed`: short seed rows reviewed outside model-generated corpus work.
- `public-import`: explicit license-gated external data with pinned revision.

## Disallowed Active Provenance

- GPT-authored corpus text.
- Kimi-authored corpus text.
- Claude-authored corpus text.
- Any `author_type=llm-curated` row in the default corpus.
- Any `author_model` matching `gpt`, `kimi`, `claude`, or `llm`.

## Quarantine Rule

- Existing tracked source packs authored by an LLM are inactive.
- Inactive packs must not be read by `prepare-corpus`.
- They may remain as reference material until a later commit deletes or rewrites
  them from approved sources.

## Artifact Rule

- Model, tokenizer, checkpoint, eval, and generated dataset artifacts trained on
  disallowed data are invalid for acceptance.
- After changing active provenance, remove old `data/train*` and `data/models/*`
  artifacts before retraining.

## Verification

```bash
PYTHONPATH=training python3 -m lkjai_train.cli prepare-corpus
PYTHONPATH=training python3 -m lkjai_train.cli validate-dataset
```

Expected: generated row metadata uses only allowed provenance and no disallowed
author model.

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
- `kimi-generated`: Kimi Code generated corpus rows from approved prompts.

## Disallowed Active Provenance

- Codex/GPT-authored corpus text.
- Claude-authored corpus text.
- Any `author_type=llm-curated` row in the default corpus.
- Any `author_model` matching `gpt`, `codex`, `claude`, or generic `llm`.

## Quarantine Rule

- Existing tracked source packs authored by Codex/GPT are inactive.
- Inactive packs must not be read by `prepare-corpus`.
- Kimi source packs may be promoted only with `kimi-generated` metadata.

## Artifact Rule

- Model, tokenizer, checkpoint, eval, and generated dataset artifacts trained on
  disallowed data or JSON actions are invalid for acceptance.
- After changing active provenance, remove old `data/train*` and `data/models/*`
  artifacts before retraining.

## 60k Corpus Source Mix

The mainline 60,000 row corpus uses this approximate provenance mix:

- `repo-derived`: ~68% (docs, source code, navigation, planning)
- `runtime-schema-derived`: ~15% (typed tools, APIs, action schemas)
- `test-derived`: ~10% (fixtures, failure diagnosis, confirmation, revision)
- `human-seed`: ~7% (preference rubrics, safety boundaries)
- `public-import`: ~0% (not available locally; redistributed to repo-derived)
- `kimi-generated`: full 500M-token corpus under `data/kimi-corpus/`

## Verification

```bash
PYTHONPATH=training python3 -m lkjai_train.cli prepare-corpus
PYTHONPATH=training python3 -m lkjai_train.cli validate-dataset
```

Expected: generated row metadata uses only allowed provenance and XML actions.
After generation, run:

```bash
rg -n 'gpt|codex|claude|llm-curated' data/train/datasets/ || true
```

This must return no matches in active split files.

# Training Data Provenance

## Goal

Keep active training data traceable to project canon, runtime contracts, tests,
or pinned permissive public pretraining sources.

## Allowed Active Provenance

- `repo-derived`: generated from `docs/`, source schemas, tests, or manifests.
- `test-derived`: generated from deterministic verification fixtures.
- `runtime-schema-derived`: generated from typed tool, API, or action schemas.
- `human-seed`: short seed rows reviewed outside model-generated corpus work.
- `public-import`: explicit license-gated external data with pinned revision;
  inactive by default for the Kimi full corpus.
- `public-pretrain`: pinned permissive English public text for causal LM
  pretraining.
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
- Kimi-only teacher-data restrictions are removed; Kimi rows are optional.

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
- `public-pretrain`: 500M-token Cosmopedia `text`-only English corpus under
  `data/public-corpus/`
- `kimi-generated`: optional SFT/tool corpus outside the pretraining target

## Public Dataset Policy

Public-import rows remain conservative and opt-in:

- Allowed active licenses: `Apache-2.0`, `MIT`, `BSD-2-Clause`, `BSD-3-Clause`.
- Placeholder revisions are rejected for active sources.
- Cosmopedia active rows must use only generated `text` and must exclude
  `prompt` and `seed_data`.
- ODC-By and CC-BY sources are reference-only under the current policy.
- OASST1 English, OASST2 English, smol-smoltalk, and Hermes Function-Calling V1
  may become active only after revision pinning and local normalization.
- Dolly and xLAM are review-only in mainline because their license or access
  terms need a separate legal decision.

## Verification

```bash
PYTHONPATH=training/package python3 -m lkjai_train.cli prepare-corpus
PYTHONPATH=training/package python3 -m lkjai_train.cli validate-dataset
```

Expected: generated row metadata uses only allowed provenance and XML actions.
After generation, run:

```bash
rg -n 'gpt|codex|claude|llm-curated' data/train/datasets/ || true
```

This must return no matches in active split files.

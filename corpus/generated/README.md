# Generated Corpus

## Purpose

Validated generated corpus artifacts that are part of the active project state
live here.

## Contents

- `kimi-full` and `kimi-sft-pilot` were deleted from the active tree
  because they trained the wrong priors and failed runtime contract fidelity.
- [kimi-sft-60m/README.md](kimi-sft-60m/README.md): next accepted
  first-party SFT corpus, with tracked train, validation, and holdout seed
  shards.
- [preference-pairs/README.md](preference-pairs/README.md): separate preference-pair artifacts,
  never active SFT rows.
- The active 500M public pretraining corpus is stored under ignored
  `data/public-corpus/`; source recipes live in `corpus/sources/`.

## Rules

- Do not commit failed, quarantined, or partial generation outputs.
- Keep manifests next to the generated corpus they describe.

# Generated Corpus

## Purpose

Validated generated corpus artifacts that are part of the active project state
live here.

## Contents

- `kimi-full-v1` and `kimi-sft-pilot-v1` were deleted from the active tree
  because they trained the wrong priors and failed runtime contract fidelity.
- `kimi-sft-60m-v2/`: next accepted first-party SFT corpus, committed only
  after runtime validator and fixture gates pass.
- `pref-v1/`: separate preference-pair artifacts, never active SFT rows.
- The active 500M public pretraining corpus is stored under ignored
  `data/public-corpus/`; source recipes live in `corpus/sources/`.

## Rules

- Do not commit failed, quarantined, or partial generation outputs.
- Keep manifests next to the generated corpus they describe.

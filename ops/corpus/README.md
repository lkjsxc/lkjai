# Corpus Ops

Owner: `ops/corpus/README.md`.
State: canonical documentation.


## Purpose

This directory contains isolated public corpus acquisition and validation
entrypoints used only by the `corpus` Compose profile.

## Contents

- [entrypoint.sh](entrypoint.sh): profile command dispatcher.
- [kimi_sft/README.md](kimi_sft/README.md): Kimi SFT validation and promotion
  command package.
- [kimi_sft_tool](kimi_sft_tool): Kimi SFT command wrapper.
- [public_pretrain_tool](public_pretrain_tool): corpus-only acquisition,
  preparation, packed-cache, and validation helper.
- [public_pretrain/README.md](public_pretrain/README.md): helper module map.

## Rules

- Keep this tooling outside the product train, serve, runtime, and verification
  paths.
- Prefer native product tooling when a capability graduates out of corpus
  acquisition.
- Split this directory's helper files before adding more behavior.

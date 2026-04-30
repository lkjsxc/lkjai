# Kimi Corpus Tool

## Purpose

This tool drives Kimi API corpus generation, validation, and scoring for
staging outputs.

## Contents

- [generate_kimi_corpus.py](generate_kimi_corpus.py): main generator CLI.
- [score_corpus.py](score_corpus.py): corpus scoring CLI.
- [launch_background.sh](launch_background.sh): background generation helper.
- [kimi_lib/](kimi_lib/): generator, prompt, scoring, and manifest modules.
- [prompts/](prompts/): Kimi prompt templates.

## Rules

- Stage outputs under `data/kimi_synthetic/` or `runs/`.
- Commit only normalized validated artifacts under `corpus/generated/`.

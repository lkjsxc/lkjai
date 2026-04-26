# Training Package

## Purpose

This directory is the Python import root for `lkjai_train`.

## Contents

- [lkjai_train/](lkjai_train/): corpus builders, tokenizer code, scratch
  training loop, evals, checkpointing, settings, and serving helpers.

## Import Contract

Use:

```bash
PYTHONPATH=training/package python -m lkjai_train.cli --help
```

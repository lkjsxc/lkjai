# Scratch Model Defaults

## Goal

Keep one coherent contract for training, export, and serving on a local RTX
3070 class machine.

## Defaults

- Default serving name: `lkjai-scratch-60m`.
- Default serving family: local scratch dense decoder.
- Default serving scale: `50M-60M` parameters.
- Default artifact root: `data/models/lkjai-scratch-60m/`.
- Default training starts from random initialization.
- Default tokenizer is a local byte-level BPE tokenizer trained on the train
  split only.

## Context Contract

- Active model context is `1024` tokens.
- Training, evaluation, export, and serving must use the same `1024` token
  contract.
- Long conversations depend on summaries, retrieval, and compact tool results,
  not a hidden larger context window.

## Precision And Runtime

- Training default: BF16 when the local CUDA stack supports it.
- Training fallback: FP16 only when BF16 is unavailable or unstable.
- Serving default: Python/Torch OpenAI-compatible runtime with KV-cache decode.
- Runtime quality must come from real generation. No supervised exact-match
  lookup is allowed in the default path.

## Environment

- `MODEL_NAME=lkjai-scratch-60m`
- `MODEL_CONTEXT_TOKENS=1024`
- `MODEL_MAX_NEW_TOKENS=512`
- `MODEL_TEMPERATURE=0.2`

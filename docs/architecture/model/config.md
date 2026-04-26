# Scratch Model Defaults

## Goal

Keep one coherent contract for training, export, and serving on a local RTX
3070 class machine.

## Defaults

- Default serving name: `lkjai-scratch-20m`.
- Default serving family: local scratch dense decoder.
- Default serving scale: about `20M` parameters for the current corpus.
- Long-term serving scale: about `60M` parameters after corpus growth.
- Default artifact root: `data/models/lkjai-scratch-20m/`.
- Default training starts from random initialization.
- Default tokenizer is a local byte-level BPE tokenizer trained on the train
  split only.
- Default training objective is `causal_lm_full`.
- XML-action SFT is available as `assistant_masked_sft`; it masks non-assistant
  labels and preserves the message serialization used by the runtime.

## Context Contract

- Active model context is `1024` tokens.
- Training, evaluation, export, and serving must use the same `1024` token
  contract.
- Long conversations depend on summaries, retrieval, and compact tool results,
  not a hidden larger context window.

## Precision And Runtime

- Training default: BF16 when the local CUDA stack supports it.
- Training fallback: FP16 only when BF16 is unavailable or unstable.
- FP16 training uses AMP gradient scaling.
- Activation checkpointing is disabled by default for `scratch-20m`; larger
  presets enable it only when the benchmarked memory policy requires it.
- Serving default: Python/Torch OpenAI-compatible runtime with KV-cache decode.
- Runtime quality must come from real generation. No supervised exact-match
  lookup is allowed in the default path.

## Environment

- `MODEL_NAME=lkjai-scratch-20m`
- `MODEL_CONTEXT_TOKENS=1024`
- `MODEL_MAX_NEW_TOKENS=512`
- `MODEL_TEMPERATURE=0.2`
- `TRAIN_MODEL_PRESET=scratch-20m`
- `TRAIN_OBJECTIVE=causal_lm_full`
- `TRAIN_EXPORT_CHECKPOINT=best`

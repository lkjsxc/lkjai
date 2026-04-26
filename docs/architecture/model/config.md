# Scratch Model Defaults

## Goal

Keep one coherent contract for training, export, and serving on a local RTX
3070 class machine.

## Defaults

- Default serving name: `lkjai-scratch-40m`.
- Default serving family: local scratch dense decoder.
- Default serving scale: about `40M` parameters for the current corpus.
- Long-term serving scale: about `60M` parameters after corpus growth and
  behavior gates justify the larger run.
- Default artifact root: `data/models/lkjai-scratch-40m/`.
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
- Activation checkpointing is enabled for `scratch-40m` only when the active
  JSON config or auto-batch policy requires it.
- Serving default: Python/Torch OpenAI-compatible runtime with KV-cache decode.
- Runtime quality must come from real generation. No supervised exact-match
  lookup is allowed in the default path.

## Environment

- `MODEL_NAME=lkjai-scratch-40m`
- `MODEL_CONTEXT_TOKENS=1024`
- `MODEL_MAX_NEW_TOKENS=512`
- `MODEL_TEMPERATURE=0.2`
- `TRAIN_CONFIG=/workspace/configs/training/scratch_40m_12h.json`
- `TRAIN_MODEL_PRESET=scratch-40m`
- `TRAIN_OBJECTIVE=causal_lm_full`
- `TRAIN_EXPORT_CHECKPOINT=best`

## 40M Agent Preset

`scratch-40m` is the active default training and serving target.

- Vocabulary: `8192`
- Context: `1024`
- Layers: `10`
- Hidden size: `576`
- Attention heads: `8`
- KV heads: `2`
- FFN size: `1536`
- Approximate parameters: `39.6M`
- Default optimizer steps: `400000`

Use the committed JSON config for Docker training. Environment variables may
override individual JSON values for smoke checks or experiments.

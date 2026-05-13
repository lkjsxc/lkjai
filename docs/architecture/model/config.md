# Scratch Model Defaults

Owner: `docs/architecture/model/config.md`.
State: canonical model defaults.

## Goal

Keep one coherent contract for training, export, and serving on a local RTX
3070 class machine.

## Defaults

- Default serving name: `decoder-40m-3070`.
- Default serving family: local scratch decoder-only Transformer.
- Default serving scale: about `40M` parameters for the current corpus.
- Long-term serving scale: about `60M` parameters after corpus growth and
  behavior gates justify the larger run.
- Default artifact root: `data/models/decoder-40m-3070/`.
- Default training starts from random initialization.
- Default tokenizer is a local byte-level BPE tokenizer trained on the train
  split only, with canonical XML-like tags added as single tokens.
- Default training objective is `causal_lm_full`.
- XML-action SFT is available as `assistant_masked_sft`; it masks non-assistant
  labels and preserves the message serialization used by the runtime.

## Acceptance Pointer

- Canonical first same-model acceptance target:
  `configs/native/decoder_40m_bf16_3070.json`.
- Canonical first acceptance training config:
  `configs/training/decoder_2h_40m_3070.json`.
- Dense 20M/40M configs remain foundation and regression evidence, not
  decoder-product acceptance evidence.

## Context Contract

- Active model context is `1024` tokens.
- Training, evaluation, export, and serving must use the same `1024` token
  contract.
- Long conversations depend on summaries, retrieval, and compact tool results,
  not a hidden larger context window.

## Precision And Runtime

- Training default: BF16 when the local CUDA stack supports it.
- Current accepted training path requires CUDA BF16 capability.
- FP16 fallback and AMP gradient scaling are backlog items, not accepted dense
  trainer behavior.
- Activation checkpointing and auto-batch are backlog items.
- Serving default: native OpenAI-compatible inference with artifact load,
  readiness, and explicit unsupported chat decode for non-decoder artifacts.
- Runtime quality must come from real generation. No supervised exact-match
  lookup is allowed in the default path.

## Environment

- `MODEL_NAME=decoder-40m-3070`
- `MODEL_CONTEXT_TOKENS=1024`
- `MODEL_MAX_NEW_TOKENS=512`
- `MODEL_TEMPERATURE=0.2`
- `TRAIN_CONFIG=/workspace/configs/training/decoder_2h_40m_3070.json`
- `TRAIN_NATIVE_CONFIG=/workspace/configs/native/decoder_40m_bf16_3070.json`
- `TRAIN_MODEL_PRESET=decoder-40m-3070`
- `TRAIN_OBJECTIVE=causal_lm_full`
- `TRAIN_EXPORT_CHECKPOINT=best`

## 40M Agent Preset

`decoder-40m-3070` is the active chat artifact target. Dense 40M remains a
diagnostic foundation, not a chat artifact.

- Vocabulary: `8192`
- Context: `1024`
- Layers: `10`
- Hidden size: `576`
- Attention heads: `8`
- KV heads: `2`
- Head dimension: `72`
- FFN size: `1536`
- Approximate parameters: `39.6M`
- Default optimizer steps: `400000`

Use the committed JSON config for Docker training. Environment variables may
override individual JSON values for smoke checks or experiments.

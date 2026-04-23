# Lightweighting Contract

## Required

- Keep the default scratch preset small enough for RTX 3070 8GB.
- Store model manifests beside scratch checkpoints.
- Prefer architecture and batch-size reductions before quantization.
- Verification may use deterministic tiny scratch artifacts.

## Optional Hooks

- Add int8 or lower-precision inference after real Rust decoding exists.
- Add GGUF export only as an optional compatibility experiment.
- Add distillation only after teacher, dataset, and license policy are written.

## Rejected For V1

- A hard 512 MiB production artifact target.
- Extreme quantization before behavioral evals are meaningful.
- Pretrained quantized serving models as the default runtime.

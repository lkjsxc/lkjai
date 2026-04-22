# Lightweighting Contract

## Required

- Serve quantized GGUF models through the model server.
- Default serving quantization is `Q4_K_M`.
- Store model manifests beside downloaded or converted models.
- Verification may use a fake model client instead of downloading large weights.

## Optional Hooks

- Add `Q5_K_M` when answer quality matters more than VRAM headroom.
- Add merged-LoRA GGUF export after QLoRA tuning is stable.
- Add int8 or fp16 only when the target GPU has enough memory.

## Rejected For V1

- A hard 512 MiB production artifact target.
- Extreme quantization before behavioral evals are meaningful.
- Distillation before the teacher, dataset, and license path are documented.

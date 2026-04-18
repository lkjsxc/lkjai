# Lightweighting Contract

## Goal

Guarantee a deployable inference artifact at `<= 512 MiB`.

## Strategy

- Keep a dense base model around `~250M` parameters.
- Publish an inference-ready 4-bit quantized artifact.
- Keep larger training checkpoints for development workflows.
- Track artifact metadata in CI/verification output.

## Verification Rule

- Release gate fails if deploy artifact exceeds `512 MiB`.
- Artifact-size check is mandatory in compose verification.

## Techniques

- Post-training quantization (4-bit).
- Optional structured pruning for margin.
- Optional distillation into smaller deploy variants.


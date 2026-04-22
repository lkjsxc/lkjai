# Inference Runtime

## Loading

- Model directory defaults to `data/train/models/lkj-150m`.
- The loader reads `config.json`, `model.safetensors`, `tokenizer.json`, and
  `manifest.json`.
- Candle is the Rust inference backend.
- CPU loading upcasts fp16 weights to fp32 when required by Candle kernels.
- CUDA loading keeps fp16 weights on device. `INFERENCE_DEVICE=cuda` fails
  explicitly if the binary or host cannot initialize CUDA.

## Generation

- Generation runs the exported decoder-only transformer.
- The web chat must never return canned model-status text as an assistant
  response.
- Missing model artifacts are reported as explicit runtime errors instead of
  being hidden behind dummy chat behavior.

## Compatibility

- Exported model config must match Rust loader expectations.
- Tokenizer files must be stored beside the model export.
- Runtime validates the co-located tokenizer hash from `manifest.json`.
- Legacy exports must be regenerated with `export-model`.

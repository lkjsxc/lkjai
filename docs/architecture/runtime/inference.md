# Inference Runtime

## Loading

- Model directory defaults to `data/models/lkj-150m`.
- The loader reads `config.json`, `model.safetensors`, and `tokenizer.json`.
- Candle is the Rust inference backend.
- CPU loading upcasts fp16 weights to fp32 when required by Candle kernels.
- CUDA loading keeps fp16 weights on device when CUDA is available.

## Generation

- Generation runs the exported decoder-only transformer.
- The web chat must never return canned model-status text as an assistant
  response.
- Missing model artifacts are reported as explicit runtime errors instead of
  being hidden behind dummy chat behavior.

## Compatibility

- Exported model config must match Rust loader expectations.
- Tokenizer files must be stored beside the model export.
- Legacy exports may be repaired by copying `data/tokenizers/tokenizer.json`
  into the model directory.

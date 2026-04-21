# Inference Runtime

## Loading

- Model directory defaults to `data/models/lkj-150m`.
- The loader reads config, tokenizer, and safetensors weights.
- Candle is the Rust inference backend.

## Generation

- Generation supports max-token, temperature, and top-p settings.
- The first implementation may return a deterministic fallback when no model
  artifact is present.
- Missing model artifacts are reported in transcripts instead of crashing the
  server.

## Compatibility

- Exported model config must match Rust loader expectations.
- Tokenizer files must be stored beside the model export.

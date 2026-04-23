# Lightweighting References

## GGUF

- GGUF export is optional future compatibility work.
- It is not the default serving path for scratch artifacts.
- Source: <https://github.com/ggml-org/llama.cpp>

## Quantization

- Quantization may be added after real scratch inference and evals exist.
- Avoid compression work before the baseline model can be trained and loaded.

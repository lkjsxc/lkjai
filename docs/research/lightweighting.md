# Lightweighting Research

## llama.cpp GGUF

- GGUF quantization is the v1 local serving path.
- `Q4_K_M` is the default balance for RTX 3070 8GB.
- Source: <https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md>

## Qwen Quantization

- Qwen documentation covers llama.cpp conversion and quantization.
- Source: <https://qwen.readthedocs.io/en/stable/quantization/llama.cpp.html>

## V1 Boundary

- The model server must load or clearly report model load errors.
- The Rust app must remain focused on orchestration and transcripts.

# Third-Party Code

Owner: `third_party/README.md`.

## cuDNN Frontend

- Path: `third_party/cudnn_frontend`.
- Source: NVIDIA `cudnn-frontend`.
- Purpose: builds decoder BF16 causal GQA SDPA graphs for the CUDA training
  acceptance path.
- Policy: keep this as a pinned git submodule. Authored wrappers live under
  `native/src/` and should stay small; do not edit vendor sources in place.

The local wrapper follows NVIDIA's attention operation documentation and SDPA
sample layout, using BF16 I/O with FP32 compute/intermediate data, top-left
causal masking, and explicit stats handoff from forward to backward.

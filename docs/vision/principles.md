# Principles

## Build Principles

- Documentation defines behavior before implementation.
- Implementation follows the smallest coherent contract that satisfies the docs.
- Agent behavior matters more than raw parameter count.
- Verification stays small enough to run routinely in Docker Compose.
- Runtime defaults must fit an RTX 3070 8GB workstation.

## Product Principles

- The web app is local-first.
- Host-YOLO actions are explicit in transcripts.
- Tool execution favors operator power over sandboxing.
- Dangerous defaults must be documented plainly.
- Multi-turn continuity requires transcript, summary, and durable memory.

## Model Principles

- The v1 model is a small dense decoder-only model.
- Prefer RMSNorm, RoPE, SwiGLU, pre-norm, and GQA-capable architectures.
- Prefer pretrained open weights plus local post-training over brute-force
  pretraining.
- Quantized local serving is accepted when the model server can load and answer.
- Custom model kernels are research, not the v1 production path.

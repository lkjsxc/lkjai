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
- The default model starts from random initialization.
- Pretrained models are rejected alternatives or future comparison baselines,
  never default serving or training dependencies.
- Agent behavior is taught through local structured trajectories, not inherited
  from an upstream chat model.
- Rust inference is the default direction for serving scratch artifacts.

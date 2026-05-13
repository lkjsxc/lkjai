# Rejected Alternatives

Owner: `docs/decisions/rejected-alternatives.md`.
State: canonical rejected decision list.

## Pretrained Default Runtime

- Rejected.
- Qwen, Gemma, Kimi, DeepSeek, and similar systems are not default runtime
  dependencies.
- They may be future comparison baselines or sources of design lessons.

## QLoRA First

- Rejected.
- Adapter training moves the research center back to pretrained behavior.
- This project intentionally studies weak local scratch models plus agent
  scaffolding.

## Pretrained Tokenizer

- Rejected for the default path.
- The tokenizer is part of the from-scratch artifact chain.
- Future baselines may compare against pretrained tokenizers explicitly.

## MoE

- Rejected.
- Dense small models are simpler to train and inspect locally.
- MoE increases implementation and routing complexity without solving current needs.

## Huge Native Context

- Rejected as the memory strategy.
- Long context is expensive on 8GB VRAM.
- External memory, retrieval, and summaries are more realistic.

## Initial Multimodality

- Rejected.
- It dilutes the core agent loop, tool use, and memory work.

## Python Default Serving

- Rejected for active serving.
- Native C++/CUDA tensor decoding is the implementation direction.
- The product path is split across static web, sandbox `/api/*`, and inference
  `/v1/*` services.

## Deterministic Inference Stub

- Rejected as a competency path.
- Artifact validation is useful health information but not model behavior.
- Generated responses from trained checkpoints are required for acceptance.

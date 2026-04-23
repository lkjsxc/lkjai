# Rejected Alternatives

## Pretrained Default Runtime

- Rejected for v1.
- Qwen, Gemma, Kimi, DeepSeek, and similar systems are not default runtime
  dependencies.
- They may be future comparison baselines or sources of design lessons.

## QLoRA First

- Rejected for v1.
- Adapter training moves the research center back to pretrained behavior.
- This project intentionally studies weak local scratch models plus agent
  scaffolding.

## Pretrained Tokenizer

- Rejected for the default path.
- The tokenizer is part of the from-scratch artifact chain.
- Future baselines may compare against pretrained tokenizers explicitly.

## MoE

- Rejected for v1.
- Dense small models are simpler to train and inspect locally.
- MoE increases implementation and routing complexity without solving v1 needs.

## Huge Native Context

- Rejected as the memory strategy.
- Long context is expensive on 8GB VRAM.
- External memory, retrieval, and summaries are more realistic.

## Phase-1 Multimodality

- Rejected for v1.
- It dilutes the core agent loop, tool use, and memory work.

## Python Default Serving

- Accepted for current v1 serving.
- Native Rust tensor decoding remains a future implementation direction.
- The web app and inference runtime must remain separate containers.

## Deterministic Inference Stub

- Rejected as a competency path.
- Artifact validation is useful health information but not model behavior.
- Generated responses from trained checkpoints are required for acceptance.

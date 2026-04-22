# Rejected Alternatives

## From-Scratch Pretraining First

- Rejected for v1.
- RTX 3070 8GB cannot realistically produce strong agent behavior this way.
- The project needs tool use, memory, and multi-turn planning sooner.

## MoE

- Rejected for v1.
- Dense 0.6B-1.7B models are simpler to tune and serve locally.
- MoE increases implementation and routing complexity without solving v1 needs.

## Huge Native Context

- Rejected as the memory strategy.
- Long context is expensive on 8GB VRAM.
- External memory, retrieval, and summaries are more realistic.

## Phase-1 Multimodality

- Rejected for v1.
- It dilutes the core agent loop, tool use, and memory work.

## Custom Production Model Runtime

- Rejected for v1.
- llama.cpp provides a more realistic local serving path.
- Rust should focus on orchestration, not model kernel parity.

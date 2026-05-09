# Evidence Records

## Purpose

Every accepted performance checkpoint needs enough context for another LLM
agent to reproduce or challenge it.

## Required Fields

- Date and git commit.
- Exact Docker Compose or native command.
- Config paths and config digests when available.
- Packed-cache, tokenizer, checkpoint, and export paths.
- GPU name, compute capability, driver, CUDA, cuDNN, and CUDA arch flags.
- Backend fields for GEMM, attention, decode, KV cache, optimizer, and report
  status.
- Timing split, throughput, peak memory, workspace sizes, and artifact checks.
- Explicit limitations and non-claims.

## Evidence Families

- Dense foundation evidence records accepted dense BF16 substrate changes.
- Decoder evidence records forward, backward, KV-cache, and serving acceptance.
- Scale profile evidence records larger GPU or multi-GPU results without
  changing the RTX 3070 gate.
- Inference acceleration evidence records optional backend comparisons after
  native decode acceptance.

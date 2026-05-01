# Kimi Prompt Templates

## Purpose

Prompt templates instruct Kimi to generate fixture-grounded pretraining and SFT
rows for local validation.

## Contents

- [pretrain_v1.txt](pretrain_v1.txt): historical pretraining prompt.
- [pretrain_v2.txt](pretrain_v2.txt): current pretraining prompt.
- [prompt_refiner.txt](prompt_refiner.txt): sample-driven prompt refinement
  prompt.
- [sft_api-v1.txt](sft_api-v1.txt): historical API SFT prompt.
- [sft_api-v2.txt](sft_api-v2.txt): current API SFT prompt.
- [sft_v1.txt](sft_v1.txt): historical CLI SFT prompt.
- [sft_v2.txt](sft_v2.txt): current CLI SFT prompt.

## Rules

- Prompts must request JSON-only output.
- SFT prompts must match the runtime tool profile and XML action schema.

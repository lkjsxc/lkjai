# Lightweighting Research

## Fujitsu

- Fujitsu lightweighting references inform future quantization, pruning,
  distillation, and NAS directions.
- Source: <https://en-documents.research.global.fujitsu.com/takane-enterprise-llm-with-generative-ai-reconstruction/>

## V1 Boundary

- v1 requires fp16 safetensors export under 512 MiB.
- v1 documents int8 and int4 export hooks without requiring production quality.

## Acceptance Rule

- A lightweight export path is accepted only when Rust serving can load or
  clearly reject the artifact with a useful error.

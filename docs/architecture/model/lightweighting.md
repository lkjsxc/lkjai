# Lightweighting Contract

## Required

- Export fp16 safetensors for v1 serving.
- Fail export if the serving artifact exceeds 512 MiB.
- Keep tokenizer and config beside weights for deterministic loading.

## Optional Hooks

- Add int8 export after fp16 serving is stable.
- Add int4 export only when verification can load and generate from it.
- Add distillation only after a teacher model and licensing path are documented.

## Future Research

- Fujitsu-style extreme quantization is a research direction.
- Structured pruning is a research direction.
- NAS-guided layer shape search is a research direction.

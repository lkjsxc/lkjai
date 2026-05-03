# Packed Cache Builder

This Rust utility builds deterministic `lkjai-packed-cache-v2` token caches from
JSONL source rows using a HuggingFace `tokenizer.json`.

It is intentionally separate from native training code so packed-cache
construction uses the canonical tokenizer implementation instead of duplicating
tokenization in C++.

## Contents

- [src/](src/): builder and validator source code.
- [Cargo.toml](Cargo.toml): crate manifest for package
  `lkjai_packed_cache_builder`.

## Dense Seq1024 Usage

```bash
docker compose --progress quiet --profile verify run --rm --entrypoint cargo verify \
  run -p lkjai_packed_cache_builder -- build \
  --source /workspace/data/train/datasets/train.jsonl \
  --tokenizer /workspace/data/train/tokenizer/tokenizer.json \
  --config /workspace/configs/native/native_dense_20m_bf16_3070.json \
  --out /workspace/data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --split train --objective causal_lm_full \
  --seq-len 1024 --sequence-count 8192 \
  --seed 20260504 --run-id dense-2h-3070
```

```bash
docker compose --progress quiet --profile verify run --rm --entrypoint cargo verify \
  run -p lkjai_packed_cache_builder -- validate \
  --cache /workspace/data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --source /workspace/data/train/datasets/train.jsonl \
  --tokenizer /workspace/data/train/tokenizer/tokenizer.json \
  --config /workspace/configs/native/native_dense_20m_bf16_3070.json
```

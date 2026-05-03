# Dense BF16 CUDA Foundation

This is the canonical operator runbook for the accepted native milestone.
Dense means the BF16 CUDA token-embedding plus LM-head path. It is not
decoder-only transformer training, autoregressive chat, or model competency.

## Scope

- Accepted: `lkjai-native-train --train --mode dense` and `--smoke`.
- Accepted: dense BF16 export inspection and single-step logits inference.
- Diagnostic: `--mode transformer`, transformer logits checks, and transformer
  artifacts.
- Unsupported: `/v1/chat/completions` autoregressive decode. The server returns
  HTTP `422` with no `choices` until the decode milestone lands.

## Build Cache

Use an existing reviewed packed-cache v2 directory, or build one through the
Rust cache builder:

```sh
cargo run -p lkjai-packed-cache-builder -- \
  --config configs/corpus/public-pretrain.json \
  --out data/train/datasets/packed/train-causal_lm_full-seq1024
```

The native trainer validates `metadata.json`, `tokens.bin`, `loss_mask.bin`,
`starts.bin`, token dtype, vocab, sequence length, token count, row count, and
start offsets once at run start. Dense training then uses a persistent packed
reader with open file descriptors and bounded per-batch reads.

## Train

```sh
DATA_DIR=data/train-dense-foundation \
MODEL_NAME=dense-foundation \
lkjai-native-train --train \
  --mode dense \
  --config configs/native/native_debug_bf16.json \
  --packed-cache data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --seq-len 1024 \
  --max-steps 128
```

Dense reports must declare `model_kind=dense`,
`accepted_cuda_training=true`, `implementation_status=accepted`,
`dense_cuda_path=true`, `loader_backend=persistent_packed_cache_reader`,
`row_layout=dense_physical_bxseq_masked_final_token`,
`matmul_plan_cache_enabled=true`, `buffer_reuse_enabled=true`, and
`timing_source=cuda_events_with_boundary_sync`. Use
`--loss-sample-interval N` or `TRAIN_LOSS_SAMPLE_INTERVAL=N` to add
deterministic trend samples to the train report.

## Inspect

```sh
lkjai-native-inspect \
  --model-dir data/train-dense-foundation/exports/dense-foundation
```

Inspection validates manifest checksums, config/tokenizer checksums, dense
weight index ranges, tensor shapes, and non-empty `weights.lkjw`.

## Dense Infer

```sh
lkjai-native-infer \
  --model-dir data/train-dense-foundation/exports/dense-foundation \
  --tokens 1,2,3
```

The command emits stable JSON with status, model kind, shape, finite flag,
checksum, top token, and CUDA capability summary. It is logits inference only.
Use `lkjai-native-logits-check` for validation/reference-check tooling.

## Benchmark

The canonical real-learning proof target is a controlled dense bigram run:

```sh
python3 tools/benchmarks/run_dense_learning_control.py \
  --run-id dense-learning-control-20260503 \
  --steps 1024 \
  --sample-interval 0.25
```

It builds a deterministic packed-cache v2 target with `seq_len=16`,
`vocab_size=256`, `row_count=128`, and cyclic token transitions over tokens
`1..64`. The run uses `native_debug_bf16`, batch size `4`, gradient
accumulation `1`, learning rate `0.001`, checkpoint interval `128`,
`run_purpose=dense_learning_control`, and `MODEL_NAME=dense-learning-control`.

Learning is demonstrated only when training succeeds, sampled losses are
finite, final loss is at least 10% below initial loss, the last-quarter sample
mean is below the first-quarter sample mean, weights changed, inspect passes,
the BF16 export/reference logits check passes, and two dense inference calls
for `--tokens 1,2,3` return matching checksums.

The 40M command remains compatibility-only:

```sh
python3 tools/benchmarks/run_dense_40m_compat.py \
  --run-id RUN_ID \
  --steps 4 \
  --sequence-count 8 \
  --sample-interval 0.25
```

The 40M command is a bounded compatibility start-check. Promotion summaries
must reject it unless a later run also satisfies accepted dense criteria:
accepted status, finite decreasing loss, valid artifacts, passing BF16
export/reference checks, `run_purpose=accepted_training` or
`run_purpose=dense_learning_control`, and positive throughput.

## Verify

```sh
cargo test --workspace
docker compose --progress quiet --profile verify run --rm verify
```

Native CTest in the verify profile covers persistent packed-cache reads,
wraparound, corrupt starts, config/cache mismatch, dense CUDA parity, dense
smoke export, dense inference, resume determinism, inspect, logits/reference
check, and explicit unsupported server decode.

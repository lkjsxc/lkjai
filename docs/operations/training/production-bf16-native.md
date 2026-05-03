# Production BF16 Native Training

## Architecture

The active accepted native CUDA path is dense BF16 CUDA. It consumes
`lkjai-packed-cache-v2` batches and exports `lkjai-native-artifact-v2`. Stable
layouts are:

- Tokens: `[B,S]` as little-endian `uint16`.
- Activations: `[B*(S-1),H]` for the current embedding-to-LM-head trainer.
- Logits: `[B*S,V]`.
- Q/K/V attention tensors are target transformer work, not current dense
  trainer behavior.

Routine verification uses `configs/native/native_debug_bf16.json`. Dense
production-like runs should use explicit dense-size configs such as
`configs/native/native_dense_20m_bf16_3070.json` or
`configs/native/native_dense_40m_bf16_3070.json`. Transformer profile configs
are separate and do not size the accepted dense parameter count.

## Precision

Serving artifacts store model tensors as BF16. Training keeps FP32 master
weights, gradients, and AdamW moments, and the CUDA forward/backward path reads
BF16 shadow tensors rebuilt from the FP32 masters. The accepted debug trainer
executes dense token embedding plus LM-head CE loss, optimizer,
checkpoint/export, and BF16 export logits probe. RMSNorm, RoPE, causal GQA
attention, and SwiGLU MLP are target transformer work.

## Packed Cache

Training requires `metadata.json`, `tokens.bin`, `loss_mask.bin`, and
`starts.bin`. The metadata `format` must be `lkjai-packed-cache-v2`, token dtype
must be `uint16`, metadata counts must match file sizes, starts must stay within
the token file, and `loss_mask` marks next-token labels that contribute to cross
entropy.

Validate the active cache with the Rust builder package
`lkjai_packed_cache_builder` before long runs. The old smoke cache at
`data/train/datasets/packed/train-causal_lm_full-seq1024` had seq16/vocab256
metadata and was invalid for seq1024 dense BF16 jobs until rebuilt.

## CLI

```sh
lkjai-native-train --train \
  --packed-cache data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --config configs/native/native_debug_bf16.json \
  --out data/train \
  --batch-size 1 \
  --seq-len 8 \
  --max-steps 2 \
  --lr 0.001 \
  --checkpoint-interval 1
```

Resume restores the checkpoint, including FP32 masters and Adam moments:

```sh
lkjai-native-train --train \
  --packed-cache data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --config configs/native/native_debug_bf16.json \
  --out data/train \
  --seq-len 8 \
  --max-steps 1 \
  --resume data/train/checkpoints/latest
```

Use `--export-artifact DIR` to write an additional serving artifact outside the
default `OUT/exports/${MODEL_NAME}` location.

Every successful smoke or train run writes `OUT/runs/train-report.json` and
prints compact JSON with schema version `3`. Dense reports identify the
precision mode as FP32 master, BF16 CUDA shadow, FP32 accumulation, and BF16
export, include `run_purpose`, and must declare
`accepted_cuda_training=true`. Transformer reports are experimental and must
declare `accepted_cuda_training=false`.

## Verification

Routine verification is:

```sh
docker compose --progress quiet --profile verify run --rm verify
```

The native CTest suite checks CUDA BF16 capability, packed-cache consumption,
finite loss, backward and AdamW weight changes, checkpoint resume, artifact
inspection, runtime loading, cache migration, and finite logits checksum.
Resume coverage verifies that resumed training matches uninterrupted dense
training for the same seed/config/dataset.

`lkjai-native-logits-check --reference-checkpoint DIR` compares an exported
dense BF16 artifact with the FP32 master checkpoint used to create it. The JSON
result reports max/mean absolute logits differences and the configured
tolerance.

The dense CUDA check must emit capability JSON with device name, compute
capability, BF16 support, cuBLASLt availability, cuDNN availability, and SDPA
eligibility for the active BF16 GQA shape.

## Verified Dense Smoke

On 2026-05-02, the native image completed a two-step dense BF16 CUDA smoke on
RTX 3070:

- command: `lkjai-native-train --smoke --steps 2`
- optimizer steps: `2`
- microsteps: `2`
- loss: `5.54436` to `5.54306`
- training logits checksum: `75e9e99a57b13691`
- dense logits-check checksum: `56f248148e361ab7`
- inspect status: `pass`
- server chat status: HTTP `422` unsupported dense autoregressive decode, with
  no `choices` field

## Bounded 40M Compatibility

Do not add this to routine verification or promotion aggregates:

```sh
python3 tools/benchmarks/run_dense_40m_compat.py \
  --run-id RUN_ID \
  --steps 4 \
  --sequence-count 8 \
  --sample-interval 0.25
```

This uses `run_purpose=bounded_compatibility_start_check`, builds a true
tokenizer-derived 8-window `seq_len=1024` cache, and checks only that the
larger dense configuration can start, checkpoint, export, and pass the logits
reference path over four optimizer steps.

## Two-Hour Dense BF16 Workflow

The production-like two-hour path is:

```sh
python3 tools/benchmarks/run_dense_2h.py \
  --run-id dense-2h-3070-$(date +%Y%m%d-%H%M%S) \
  --native-config configs/native/native_dense_20m_bf16_3070.json \
  --source data/train/datasets/train.jsonl \
  --tokenizer data/train/tokenizer/tokenizer.json \
  --cache data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --seq-len 1024 --sequence-count 8192 \
  --batch-size 1 --grad-accum 8 \
  --lr 0.0003 --pilot-steps 128 \
  --target-seconds 7200 --full
```

The runner writes GPU identity, driver/runtime, CUDA arch flags, config/cache
digests, loss samples, throughput, checkpoint/export paths, logits checks, and
limitations into the benchmark bundle.

## Limitations

Autoregressive chat/KV decode is not part of this slice. Transformer mode is
host/reference and experimental until device-resident forward/backward kernels
replace the current implementation. cuDNN SDPA is target transformer work, not
accepted dense training behavior.

# Production BF16 Native Training

## Architecture

The native dense path consumes `lkjai-packed-cache-v2` batches and exports
`lkjai-native-artifact-v2`. Stable layouts are:

- Tokens: `[B,S]` as little-endian `uint16`.
- Activations: `[B*S,H]`.
- Q/K/V: `[B,S,heads,head_dim]`.
- Logits: `[B*S,V]`.

Routine verification uses `configs/native/dense_debug_bf16.json`. The 40M shape
in `configs/native/dense_40m_bf16.json` is for manual smoke and production-like
runs only.

## Precision

Serving artifacts store model tensors as BF16. Training keeps FP32 master
weights, gradients, and AdamW moments in checkpoints. The current accepted
debug trainer executes a tensor-backed embedding and LM-head optimization slice;
the transformer projection tensors are exported in the artifact schema and are
reserved for the full CUDA transformer kernel path.

## Packed Cache

Training requires `metadata.json`, `tokens.bin`, `loss_mask.bin`, and
`starts.bin`. The metadata `format` must be `lkjai-packed-cache-v2`, token dtype
must fit the configured vocabulary, and `loss_mask` marks next-token labels that
contribute to cross entropy.

## CLI

```sh
lkjai-native-train --train \
  --packed-cache data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --config configs/native/dense_debug_bf16.json \
  --out data/train \
  --batch-size 1 \
  --seq-len 8 \
  --max-steps 2 \
  --lr 0.001 \
  --checkpoint-interval 1
```

Resume keeps optimizer step numbering:

```sh
lkjai-native-train --train \
  --packed-cache data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --config configs/native/dense_debug_bf16.json \
  --out data/train \
  --seq-len 8 \
  --max-steps 1 \
  --resume data/train/checkpoints/latest
```

Use `--export-artifact DIR` to write an additional serving artifact outside the
default `OUT/exports/${MODEL_NAME}` location.

## Verification

Routine verification is:

```sh
docker compose --progress quiet --profile verify run --rm verify
```

The native CTest suite checks CUDA BF16 capability, packed-cache consumption,
finite loss, backward and AdamW weight changes, checkpoint resume, artifact
inspection, runtime loading, and a non-empty logits checksum.

## Manual 40M Smoke

Do not add this to routine verification:

```sh
lkjai-native-train --train \
  --packed-cache data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --config configs/native/dense_40m_bf16.json \
  --out data/train-40m-smoke \
  --batch-size 1 \
  --seq-len 8 \
  --max-steps 1
```

## Limitations

The custom CUDA causal-attention fallback and cuBLASLt transformer projections
are still deferred behind the exported tensor contract. cuDNN SDPA remains
deferred until cudnn-frontend headers are available in the native image.

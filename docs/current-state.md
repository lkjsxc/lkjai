# Current State

Owner: `docs/current-state.md`.
State: canonical orientation map.

## Accepted Foundation

Dense BF16 CUDA training is the accepted native substrate. It covers
embedding and LM-head training, FP32 master weights and AdamW moments, BF16
device shadows, packed-cache input, checkpoint/export, logits checks, and
benchmark continuity. Dense evidence does not prove decoder block training or
chat-quality serving.

Product training, serving, runtime, verification, and benchmark paths remain
native C++/CUDA. Python is limited to corpus acquisition and other non-product
preparation work.

## Status Table

| Lane | State | Accepted Evidence | Blocked Capability |
|---|---|---|---|
| `dense` | accepted foundation and immediate demo target | BF16 CUDA train, checkpoint/export, logits checks, packed-cache IO, local top-k demo | decoder blocks and KV-cache decode |
| `decoder` | product acceptance target | tied artifacts, tokenizer copy, partial host-reference choices, full-decoder CUDA report contract | real block backward, optimizer coverage, and CUDA KV-cache decode |
| `transformer` | diagnostic lane | host/reference checks and probe reports | not an accepted training or serving target |

## Decoder Limits

The `decoder` model kind is the product target. Accepted reports must use:

- `implementation_status=accepted`
- `decoder_cuda_slice=full_decoder`
- `decoder_backward_backend=cuda_full_decoder`
- `kv_cache_backend=cuda_contiguous_bf16`
- `decode_backend=cuda_kv_cache`

Partial reports remain non-claims and must set `decode_supported=false`.
Host-reference recompute decode may produce decoder `choices`, but it is not
accepted CUDA KV-cache serving evidence.

The current decoder CUDA scaffolding is useful implementation work, not
accepted evidence. It must remain non-accepted while block gradients are
synthetic, optimizer coverage is not tied to real decoder backward, and decode
still calls host `transformer_forward` instead of consuming CUDA KV-cache
attention state.

## Data Readiness

Decoder acceptance is also blocked on real local data inputs. A fresh checkout
does not contain `data/train/tokenizer/tokenizer.json`, and any seq1024 cache
path must be treated as unproven until strict validation shows it was built
from public-pretrain JSONL with the decoder tokenizer and
`configs/native/decoder_40m_bf16_3070.json`.

Required data-prep order:

```bash
docker compose --profile corpus run --build --rm corpus build-tokenizer
docker compose --profile corpus run --rm corpus validate-public-pretrain
docker compose --profile corpus run --rm corpus build-public-pretrain-cache
docker compose --profile corpus run --rm corpus \
  lkjai-native-packed-cache validate \
    --cache /app/data/train/datasets/packed/train-causal_lm_full-seq1024 \
    --source /app/data/public-corpus/train \
    --tokenizer /app/data/train/tokenizer/tokenizer.json \
    --config /workspace/configs/native/decoder_40m_bf16_3070.json
```

The tokenizer builder writes a deterministic byte-level BPE-compatible
`tokenizer.json` whose canonical XML-like prompt and action tags are atomic
tokens. The packed-cache builder accepts one JSONL file or a directory of
sorted `*.jsonl` shards and streams rows instead of loading the full source.

## Do Not Claim

- Partial decoder CUDA is not accepted decoder CUDA training.
- Tied embedding or LM-head updates are not decoder block training.
- `host_reference_recompute` decode is not accepted CUDA KV-cache serving.
- Larger GPU profile results do not relax the RTX 3070 acceptance lane.
- A seq1024 path name is not evidence of a real seq1024 public cache; strict
  packed-cache validation must pass against the exact source, tokenizer, and
  decoder config.

Accepted decoder reports must prove real block-weight updates, full block
backward, FP32 optimizer coverage for every trainable decoder tensor,
export/logits/server checks, finite loss, passing logits checks, positive
steps and loss-token counts, contiguous CUDA BF16 KV-cache decode, zero
steady-state token allocations, and supported decode.

## Immediate Target

The immediate product target is a dense 40M native browser demo on the merged
server:

- native config: `configs/native/native_dense_40m_bf16_3070.json`
- training config: `configs/training/dense_12h_40m_3070.json`
- browser page: `GET /`
- local APIs: `GET /api/dense/status` and `POST /api/dense/next-token`
- evidence: bounded pilot checks, deterministic checksum, logits/top-k output,
  and truthful unsupported chat decode

This target does not claim autoregressive chat. It uses dense logits and top-k
output to make the accepted dense substrate visible and testable from a browser.

## Chat Target

The next chat acceptance target is the tied 40M decoder on RTX 3070:

- native config: `configs/native/decoder_40m_bf16_3070.json`
- training config: `configs/training/decoder_2h_40m_3070.json`
- required report fields include `implementation_status=accepted`,
  `accepted_cuda_training=true`, `decoder_cuda_slice=full_decoder`,
  `decoder_block_weight_changed=true`,
  `decoder_backward_backend=cuda_full_decoder`,
  `kv_cache_backend=cuda_contiguous_bf16`, and
  `decode_backend=cuda_kv_cache`

The two-hour RTX 3070 target is the acceptance lane documented by the training
config. Code should validate truth fields and config shape, not treat
`target_seconds > 0` as a promotion shortcut.

## Research Synthesis

The latest deep research report under `tmp/deep-research-report (54).md`,
modified `2026-05-12`, supports this order: promote the dense 40M path into a
real native browser demo first, keep decoder chat as the later acceptance
target, finish contiguous BF16 KV-cache decode before chat claims, and only
then broaden performance, batching, frontend, and multi-GPU work.

Durable conclusions now owned by docs:

- Distributed training order: tensor parallelism, activation checkpointing,
  pipeline staging, communication overlap, then optimizer sharding.
- Large-profile gates: memory accounting, dataset lineage, and profile-only
  status until the local decoder lane passes.
- KV-cache decode gates: allocation accounting and stop-token behavior.
- Evidence package pattern: dated tracked evidence page plus generated
  benchmark manifest under ignored `artifacts/`.
- Kimi SFT flow: generate into quarantine, validate schema/provenance/replay,
  then promote only passing shards into `corpus/generated/kimi-sft-60m`.
- Promoted-run bundle: include train report, metrics, plots, GPU capability,
  Nsight reports, config, tokenizer digest, dataset manifest, and transcript.
- Dense demo order: docs contract, dense local APIs, root browser page,
  checksum/top-k contract tests, bounded pilot evidence.
- Serving order: request validation, prompt serialization, prefill, native
  BF16 KV-cache decode, sampler, structured metrics, then optional `kjxlkj`
  tool calls.
- Kernel policy: keep GEMMs in cuBLASLt, use correctness-first custom CUDA for
  RMSNorm, RoPE, attention glue, KV writes/reads, sampling, and cache
  bookkeeping, and consider cuDNN SDPA only after active-shape parity exists.

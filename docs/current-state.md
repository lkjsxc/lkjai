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
| `dense` | accepted foundation and browser diagnostics | BF16 CUDA train, checkpoint/export, logits checks, packed-cache IO, local top-k demo | decoder blocks and KV-cache decode |
| `decoder` | product acceptance target | tied artifacts, tokenizer copy, full-decoder train report fields, CUDA KV-cache route disclosure contract | accepted two-hour evidence on the RTX 3070 lane |
| `runtime` | native agent loop | XML actions, transcript persistence, `agent.finish`, `agent.think`, `fs.list`, and `fs.read` | memory, resource, shell, and confirmation execution |
| `transformer` | diagnostic lane | host/reference checks and probe reports | not an accepted training or serving target |

## Decoder Limits

The `decoder` model kind is the product target. Accepted reports must use:

- `implementation_status=accepted`
- `decoder_cuda_slice=full_decoder`
- `decoder_backward_backend=cuda_full_decoder`
- `kv_cache_backend=cuda_contiguous_bf16`
- `decode_backend=cuda_kv_cache`

Partial or historical training reports remain non-claims and must avoid
accepted backend names. Decoder route responses may still disclose
`cuda_reference_kv_cache` when the served artifact lacks accepted runtime
evidence. Accepted route disclosure requires the executed CUDA KV-cache path,
accepted sidecar fields, an adjacent accepted train report, and the loaded
40M RTX 3070 decoder shape.

The current decoder implementation target is the accepted path: full decoder
training state updates, optimizer moments for every trainable tensor,
checkpoint/export coverage, logits checks, and CUDA KV-cache generation. The
current code stage is still experimental because training uses host-reference
forward/backward with CUDA probes, and serving uses non-accepted decode
disclosure. The blockers are real device-resident full decoder backward,
accepted CUDA KV-cache route evidence, and generated two-hour evidence from the
documented RTX 3070 acceptance lane. Smaller or random decoder tests prove
plumbing only.

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

- Historical partial decoder CUDA is not accepted decoder CUDA training.
- Tied embedding or LM-head updates are not decoder block training.
- `cuda_reference_kv_cache` decode is not accepted CUDA KV-cache serving.
- Larger GPU profile results do not relax the RTX 3070 acceptance lane.
- A seq1024 path name is not evidence of a real seq1024 public cache; strict
  packed-cache validation must pass against the exact source, tokenizer, and
  decoder config.

Accepted decoder reports must prove block-weight updates, full block backward,
FP32 optimizer coverage for every trainable decoder tensor, export/logits/server
checks, finite loss, passing logits checks, positive steps and loss-token
counts, contiguous CUDA BF16 KV-cache decode, zero steady-state token
allocations, and supported decode.

## Dense Diagnostic Surface

The dense 40M native browser diagnostics run on the merged server:

- native config: `configs/native/native_dense_40m_bf16_3070.json`
- training config: `configs/training/dense_40m_accepted_3070.json`
- browser page: `GET /`
- local APIs: `GET /api/dense/status` and `POST /api/dense/next-token`
- evidence: bounded pilot checks, deterministic checksum, logits/top-k output,
  train-report provenance, and truthful unsupported chat decode

This surface does not claim autoregressive chat. It uses dense logits and top-k
output to keep the accepted dense substrate visible and testable from a browser
while decoder-core work proceeds.

## Active Implementation Target

The active implementation target is the tied 40M decoder on RTX 3070:

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

The latest deep research report under `tmp/deep-research-report (58).md`,
modified `2026-05-13`, was written against repository snapshot `b23be0f`.
The report supports the active order: hard-fence historical partial decoder
claims, land accepted native CUDA decode and full-decoder training evidence,
and only then broaden streaming, batching, frontend, and multi-GPU work.

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
- Dense diagnostics order: keep local APIs, root browser diagnostics,
  checksum/top-k contract tests, and bounded pilot evidence green.
- Serving order: request validation, prompt serialization, prefill, native
  BF16 KV-cache decode, sampler, structured metrics, then optional `kjxlkj`
  tool calls.
- Kernel policy: keep GEMMs in cuBLASLt, use correctness-first custom CUDA for
  RMSNorm, RoPE, attention glue, KV writes/reads, sampling, and cache
  bookkeeping, and consider cuDNN SDPA only after active-shape parity exists.

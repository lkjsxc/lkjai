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
| `dense` | accepted foundation | BF16 CUDA train, checkpoint/export, logits checks, packed-cache IO | chat-capable decoder blocks and KV-cache decode |
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

## Do Not Claim

- Partial decoder CUDA is not accepted decoder CUDA training.
- Tied embedding or LM-head updates are not decoder block training.
- `host_reference_recompute` decode is not accepted CUDA KV-cache serving.
- Larger GPU profile results do not relax the RTX 3070 acceptance lane.

Accepted decoder reports must prove real block-weight updates, full block
backward, FP32 optimizer coverage for every trainable decoder tensor,
export/logits/server checks, finite loss, passing logits checks, positive
steps and loss-token counts, CUDA KV-cache decode, and supported decode.

## Next Target

The next product acceptance target is the tied 40M decoder on RTX 3070:

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

The latest deep research report under `tmp/deep-research-report (48).md`
supports this order: keep the dense substrate as a harness, ship the tied 40M
RTX 3070 decoder target, finish contiguous BF16 KV-cache decode, and only then
broaden performance, frontend, and multi-GPU work.

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

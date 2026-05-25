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
| `runtime` | split web, sandbox, and inference services | XML actions, transcript persistence, confirmation events, read-only filesystem tools, and kjxlkj resource tools | shell tools and workspace mutation policies |
| `transformer` | diagnostic lane | host/reference checks and probe reports | not an accepted training or serving target |

## Decoder Map

The decoder lane is the first product acceptance target. Current truth is:

- Full decoder forward, CE loss, logits capture, grad-logits, and chain-rule
  backward run on CUDA from recorded decoder tape tensors.
- Decoder registry gradients are reported as `cuda_full_decoder` with
  `decoder_gradient_source=cuda_device`; host backward is parity-only evidence.
- Registry CUDA AdamW updates device FP32 masters, Adam moments, gradients, and
  BF16 shadows for decoder tensors.
- Attention uses `cuda_causal_gqa_bf16_reference` only for diagnostic smoke and
  oracle evidence. Accepted decoder training requires runtime cuDNN SDPA
  forward/backward counters, zero reference fallback counters, and at least one
  passing decoder parity sample.
- Decode owns a cached `DecoderCudaInferenceSession` per loaded artifact and
  reuses one-token buffers in the steady token loop. KV-cache route disclosure
  remains partial until accepted route evidence exists.

Detailed decoder acceptance blockers are owned by
[training.md](architecture/native/decoder/training.md),
[acceptance.md](architecture/native/decoder/acceptance.md),
[backward.md](architecture/native/decoder/backward.md),
[attention.md](architecture/native/decoder/attention.md), and
[kv-cache.md](architecture/native/decoder/kv-cache.md).

Partial reports must keep `accepted_cuda_training=false` and avoid accepted
attention or decode names. Accepted route disclosure requires accepted train
report evidence, an accepted sidecar, the loaded 40M RTX 3070 decoder shape,
  an executed CUDA KV-cache path, and
  `data/train/runs/decoder-40m-3070-route-transcript.json`.

The public pretrain cache remains local under `data/` after strict validation.
The tracked `corpus/generated/kimi-sft-60m` tree is only a tiny seed fixture.
Full Kimi SFT output is quarantined under `data/corpus/quarantine/kimi-sft-60m`
and promoted only to ignored `data/corpus/generated/kimi-sft-60m`.

## Data Map

Decoder data-prep details are owned by
[packed-cache.md](architecture/training/data/packed-cache.md),
[source-corpus.md](architecture/training/data/source-corpus.md), and
[tokenizer.md](architecture/training/data/tokenizer.md). A path named
`seq1024` is not evidence; strict packed-cache validation must prove the
source, tokenizer, config, sequence length, and checksums.

## Dense Map

Dense artifacts remain diagnostics and training artifacts only. The dense 40M
surface proves BF16 CUDA foundation behavior, checkpoint/export/logits checks,
packed-cache IO, and truthful unsupported chat decode. It does not claim
autoregressive chat.

## Acceptance Map

The active implementation target is the tied 40M decoder on RTX 3070:

- native config: `configs/native/decoder_40m_bf16_3070.json`
- training config: `configs/training/decoder_2h_40m_3070.json`
- serving artifact target: `data/models/decoder-40m-3070`

The two-hour RTX 3070 target is the acceptance lane. Code should validate truth
fields and config shape, not treat `target_seconds > 0` as a promotion shortcut.

The separate four-hour chat-attempt lane intentionally overwrites the same
serving name, `decoder-40m-3070`, with a non-acceptance
`assistant_masked_sft` artifact:

- training config:
  `configs/training/decoder_4h_chat_attempt_3070.json`
- packed cache:
  `data/train/datasets/packed/train-assistant_masked_sft-seq128`
- sequence length: `128`
- latest-checkpoint cadence: `64` optimizer steps
- run purpose: `chat_attempt`
- wall-clock target: `14400` seconds
- expected serving disclosure: `lkjai_decode_accepted=false`

Its success criterion is browser reachability to the real model path, not
answer quality or acceptance promotion. The web page may show assistant content
or a concrete failure `stop_reason` such as `invalid_action` or `model_error`.

The completed four-hour chat attempt is infrastructure evidence only. It did
not produce usable conversational behavior: direct decode returned repeated
`n` tokens under greedy sampling, while the sandbox agent rejected the raw text
as `invalid_action` because it was not a valid action block. Treat the current
artifact as proof that training, export, serving, and browser challenge paths
run end to end, not as proof that post-training improved chat quality.

Current quality blockers are likely in the data and decode interface rather
than a single missing runtime switch: the promoted SFT set is small, the run
used `seq_len=128`, the model is trained almost from scratch, and the sandbox
expects structured action XML while the decoder emits raw token text. Future
work should isolate tokenizer coverage, prompt formatting, loss-mask alignment,
and repetition control before treating longer SFT runs as quality evidence.

Research synthesis is owned by
[native-decoder-plan.md](research/native-decoder-plan.md). Reports under
`tmp/` are source inputs only, not tracked contract owners.

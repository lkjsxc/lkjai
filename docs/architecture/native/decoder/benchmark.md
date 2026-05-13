# Decoder Benchmark

## Smoke Gate

The decoder smoke gate must run through Docker Compose and prove:

- config validation,
- at least two optimizer steps,
- finite loss,
- positive non-embedding and decoder-block weight change,
- checkpoint and export artifacts,
- inspect success,
- logits check success,
- native server route success with truthful accepted or non-accepted decode
  disclosure.

## Two-Hour Gate

The accepted same-model demo requires a full CUDA BF16 decoder backend and one
RTX 3070 run using native wall-clock stop:

```bash
docker compose --profile train run --rm train \
  --train --mode decoder \
  --config /workspace/configs/native/decoder_40m_bf16_3070.json \
  --packed-cache /app/data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --seq-len 1024 --target-seconds 7200
```

`scripts/codex/run_e2e_decoder_demo.sh` must fail under
`REQUIRE_ACCEPTED_CUDA=1` unless the generated report contains accepted
full-decoder training and CUDA KV-cache decode fields. A dry-run script,
foundation server contract, or embedding/head-only CUDA slice is not accepted
evidence.

Current decoder smoke still exercises the experimental host-reference fallback
with CUDA forward probes. It must report non-accepted backend names such as
`decoder_cuda_slice=cuda_forward_probe_host_training`,
`decoder_backward_backend=host_reference`, and
the non-accepted host-reference decode backend. It is regression evidence only; the
tracked acceptance lane is still the two-hour RTX 3070 run after real CUDA
forward, backward, optimizer, logits/export/server checks, and KV-cache decode
execute.

Use smoke mode for current CUDA work:

```bash
docker compose --profile train run --rm train \
  --train --mode decoder \
  --config /workspace/configs/native/decoder_debug_bf16.json \
  --packed-cache /app/data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --seq-len 1024 --max-steps 2
```

## Evidence

Record under ignored `artifacts/benchmarks/<run-id>/`:

- train report,
- benchmark summary JSON,
- Markdown evidence report,
- exact Docker command,
- git commit,
- GPU, driver, CUDA, cuDNN, and CUDA architecture flags,
- cuBLASLt and cuDNN backend selections,
- checkpoint/export/logits checksums,
- `/v1/chat/completions` transcript smoke.
- report fields:
  `implementation_status=accepted`,
  `accepted_cuda_training=true`,
  `decoder_cuda_slice=full_decoder`,
  `attention_backend=cuda_causal_gqa_bf16_reference`,
  `decoder_backward_backend=cuda_full_decoder`,
  `kv_cache_backend=cuda_contiguous_bf16`, and
  `decode_backend=cuda_kv_cache`.

Tracked accepted evidence pages are added only after the generated train report,
artifact inspection, logits check, served artifact manifest, route transcript,
positive KV prefill bytes, and zero steady-state token allocation evidence
exist.

Tracked docs may summarize curated accepted results after the generated
artifacts exist.

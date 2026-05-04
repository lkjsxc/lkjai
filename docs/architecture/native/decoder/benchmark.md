# Decoder Benchmark

## Smoke Gate

The decoder smoke gate must run through Docker Compose and prove:

- config validation,
- at least two optimizer steps,
- finite loss,
- nonzero non-embedding weight change,
- checkpoint and export artifacts,
- inspect success,
- logits check success,
- native server chat `choices` success.

## Two-Hour Gate

The accepted same-model demo requires a full CUDA BF16 decoder backend and one
RTX 3070 run using native wall-clock stop:

```bash
python3 tools/benchmarks/run_decoder_2h.py \
  --run-id decoder-2h-3070-$(date +%Y%m%d-%H%M%S) \
  --native-config configs/native/decoder_18m_bf16_3070.json \
  --target-seconds 7200 \
  --full
```

Commit `a806c88` makes `--full` and `--require-accepted-cuda` run an acceptance
probe first. The runner must fail when the report is P0/reference or partial
CUDA. A dry-run script, P0 server contract, or embedding/head-only CUDA slice is
not accepted evidence.

Use smoke mode for current partial CUDA work:

```bash
python3 tools/benchmarks/run_decoder_2h.py \
  --run-id decoder-smoke-$(date +%Y%m%d-%H%M%S) \
  --native-config configs/native/decoder_18m_bf16_3070.json \
  --cache data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --seq-len 1024 \
  --smoke-steps 2
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

Tracked docs may summarize curated accepted results after the generated
artifacts exist.

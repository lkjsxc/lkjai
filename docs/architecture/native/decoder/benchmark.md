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

The accepted same-model demo requires one RTX 3070 run using native wall-clock
stop:

```bash
python3 tools/benchmarks/run_decoder_2h.py \
  --run-id decoder-2h-3070-$(date +%Y%m%d-%H%M%S) \
  --native-config configs/native/decoder_18m_bf16_3070.json \
  --target-seconds 7200 \
  --full
```

The runner must execute the full run when `--full` is present. A dry-run script
is not accepted evidence.

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

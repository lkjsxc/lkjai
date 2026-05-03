# Dense BF16 CUDA Training Report

This record captures the RTX 3070 dense BF16 CUDA acceptance run produced on
2026-05-04.

## Status

- Run id: `dense-2h-3070-20260504-040649`.
- Status: `pass`; full status: `completed`.
- Workflow: `tools/benchmarks/run_dense_2h.py --full`.
- Native config: `configs/native/native_dense_20m_bf16_3070.json`.
- Cache: `data/train/datasets/packed/train-causal_lm_full-seq1024`.
- Cache validation: `sequence_count=8192`, `token_count=8388608`, packed
  digest `2ffedab50c68fbdba9a5b1e7123f3ddf00a5a36a71eb0fcf487604306d2785ff`.
- GPU: `NVIDIA GeForce RTX 3070`; driver `13010`; CUDA runtime `12080`.
- CUDA arch flags: `86-real,86-virtual`.
- Result bundle:
  `artifacts/benchmarks/dense-2h-3070-20260504-040649/dense_2h_bf16_cuda/repeat-01`.

## Gate Fix Evidence

The previous dense gate checked only `tok_embeddings[0]`. On the real packed
cache, the first row does not exercise token `0`, while `lm_head` optimizer
state changed. The gate now compares both trainable FP32 master tensors:
`tok_embeddings` and `lm_head`.

- Smoke: `dense-smoke-gate-check` exited `0`; `weight_change.status=pass`.
- One-step runner gate: `dense-2h-gate-smoke-20260504-035923` returned
  `runner_status=pass`.
- Full run: `weight_changed=true`; `weight_change.status=pass`;
  `changed_tensors=2/2`; `changed_elements=20971520/20971520`;
  `max_abs_delta=2.47961`; `mean_abs_delta=0.0878965`.

## Calibration

- Pilot steps: `128`; microsteps: `1024`.
- Input tokens: `1048576`; loss tokens: `1047552`.
- Initial loss: `9.01073`; final pilot loss: `7.41296`.
- Tokens/sec: `92117.2`; median step seconds: `0.08893046875`.
- Calibrated full target: `80962` optimizer steps.
- Logits reference check: `pass`; max abs diff `0.0002928`, tolerance `0.01`.
- Weight-change check: `pass` for both trainable FP32 master tensors.

## Full Run

- Optimizer steps: `80962`; microsteps: `647696`.
- Input tokens: `663240704`; loss tokens: `662593008`.
- Elapsed seconds: `3323.27`; wall seconds: `3326.677818345`.
- Tokens/sec: `199575`; median step seconds: `0.041047281440675876`.
- Initial loss: `9.01074`; final loss: `1.70369`.
- Best loss: `1.58339` at step `80763`.
- Loss delta: `7.30704`; decrease fraction: `0.810926`.
- Learning status: `learning`.
- Logits reference check: `pass`; max abs diff `0.000706427`,
  tolerance `0.01`.
- Checkpoint checksum: `e818837f071feec2`.
- Export checksum: `e818837f071feec2`.
- Final export:
  `data/perf-runs/dense-2h-3070-20260504-040649/dense_2h_bf16_cuda/full/exports/dense-2h-20m-3070`.

## Limitations

- Single GPU only.
- Dense trainable surface is `tok_embeddings` plus `lm_head`.
- `/v1/chat/completions` remains unsupported for dense artifacts: HTTP `422`
  unsupported autoregressive decode with no `choices` field.

## Reproduction

```bash
python3 tools/benchmarks/run_dense_2h.py \
  --run-id dense-2h-3070-$(date +%Y%m%d-%H%M%S) \
  --image lkjai-native-bench --no-build --skip-cache-build \
  --native-config configs/native/native_dense_20m_bf16_3070.json \
  --source data/train/datasets/train.jsonl \
  --tokenizer data/train/tokenizer/tokenizer.json \
  --cache data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --seq-len 1024 --sequence-count 8192 \
  --batch-size 1 --grad-accum 8 \
  --lr 0.0003 --pilot-steps 128 \
  --target-seconds 7200 --full
```

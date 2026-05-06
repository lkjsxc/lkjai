# Dense Accepted Training 20260503

Canonical run id: `dense-accepted-training-20260503`

## Result

- promotion status: `promoted`
- run purpose: `accepted_training`
- report kind: `accepted_training`
- model: `dense-accepted-training`
- config: `configs/native/native_accepted_dense_bf16.json`
- Docker image: `lkjai-native-bench`
- GPU: `NVIDIA GeForce RTX 3070`
- CUDA runtime: `12080`
- cuDNN: `90800`

## Command

```sh
docker compose --profile train run --rm train \
  --train --mode dense \
  --config /workspace/configs/native/native_accepted_dense_bf16.json \
  --packed-cache /app/data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --seq-len 128 --max-steps 1024 --loss-sample-interval 64
```

The runner used the default selected learning rate `0.001`, batch size `4`,
gradient accumulation `1`, checkpoint interval `128`, loss sample interval
`64`, and sequence length `128`.

## LR Sweep

The 256-step sweep was run in order:

- `0.0003`: rejected for LR selection; final loss dropped only `0.0613058`.
- `0.001`: selected; final loss dropped `0.476085` and inspect, logits
  reference, and repeated inference checks passed.
- `0.003`: passed, but was not selected because `0.001` was the lowest passing
  learning rate.

## Dataset And Cache

The accepted run built packed-cache data from:

- source: `data/train/datasets/train.jsonl`
- tokenizer: `data/train/tokenizer/tokenizer.json`
- sequence count: `256`
- token count: `32768`
- row count: `256`
- examples consumed by builder: `541`
- seed: `20260503`
- source digest:
  `dbdf2307e9b0fb8c3c68738c13e3524ff46eccde53f83e858c738b846eb4f710`
- tokenizer digest:
  `625876ff4c76aab91d1cdc264bbfd4f6b5e8c5babe8084c4431b02c7140edf09`
- config digest:
  `b200a4ec2f90621467c1738a18a4d595b0870ed18d5e453128150d04092c0dad`
- packed checksum:
  `bea680d24aaa26803c0acf1722fed5d74d33213f0ccf28141644fcc4738a8e83`

## Loss And Tokens

- optimizer steps: `1024`
- microsteps: `1024`
- tokens seen: `524288`
- loss tokens: `520192`
- initial loss: `9.01045`
- final loss: `2.1923`
- best loss: `1.84861` at step `967`
- loss decrease fraction: `0.756694`
- first-quarter sampled mean: `8.10081`
- last-quarter sampled mean: `2.31444`
- learning status: `learning`
- throughput: `91492.4` tokens/s
- wall elapsed: `7.073612189000414` seconds

Sampled losses:

```text
1: 9.01045
64: 8.97748
128: 8.20372
192: 6.21162
256: 4.72071
320: 4.02706
384: 3.62888
448: 3.34762
512: 3.1265
576: 2.94431
640: 2.79159
704: 2.66164
768: 2.54813
832: 2.44573
896: 2.352
960: 2.26772
1024: 2.1923
```

## Checks

- checkpoint checksum: `108be3af9050f523`
- export checksum: `108be3af9050f523`
- logits checksum: `1e57a0498a8eccd3`
- inspect logits checksum: `b80be28b8e35266`
- BF16 reference check: `pass`
- max abs diff: `0.000575557`
- mean abs diff: `0.000126441`
- tolerance: `0.01`
- infer checksums: `1e57a0498a8eccd3`, `1e57a0498a8eccd3`
- top token for `--tokens 1,2,3`: `217`

Artifacts:

- summary:
  `artifacts/benchmarks/dense-accepted-training-20260503/dense_accepted_training_1024/repeat-01/accepted-training-summary.json`
- data:
  `data/perf-runs/dense-accepted-training-20260503/dense_accepted_training_1024/repeat-01/`
- export:
  `data/perf-runs/dense-accepted-training-20260503/dense_accepted_training_1024/repeat-01/exports/dense-accepted-training`

## Caveats

This is a dense embedding-plus-LM-head proof on real tokenizer-built packed
data. It does not promote the 40M target, transformer training, autoregressive
decode, or chat competency.

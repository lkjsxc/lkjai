# Dense Learning Control: 2026-05-03

## Result

Controlled dense BF16 CUDA learning is proven for the dense embedding plus
LM-head path. This does not claim transformer training, autoregressive decode,
or chat competency.

- run id: `dense-learning-control-20260503`
- command family: native packed-cache build plus
  `lkjai-native-train --train --mode dense --max-steps 1024`
- image used: `lkjai-dense-learning-control:dense-learning-control-20260503`
- run purpose: `dense_learning_control`
- model: `dense-learning-control`
- config: `configs/native/native_debug_bf16.json`
- sequence/batch/grad: `seq_len=16`, `batch_size=4`, `grad_accum=1`
- learning rate: `0.001`
- loss sample interval: `64`
- checkpoint interval: `128`

## Evidence

- status: `success`
- promotion status: `promoted`
- optimizer steps: `1024`
- tokens seen: `65536`
- initial loss: `5.54518`
- final loss: `0.101644`
- loss decrease fraction: `0.98167`
- best loss: `0.0909017` at step `1013`
- first-quarter sampled mean: `5.36123`
- last-quarter sampled mean: `0.145813`
- learning status: `learning`
- throughput: `107957` tokens/sec
- checkpoint checksum: `49e31c817db55f5`
- export checksum: `49e31c817db55f5`
- logits checksum: `8605ce9d572fbfd8`
- BF16 reference max diff/tolerance: `0.00897038 / 0.01`
- repeated dense infer checksums: `8605ce9d572fbfd8`, `8605ce9d572fbfd8`
- learning rejections: none
- promotion errors: none

The sampled losses were monotonic at the configured interval: `5.54518`,
`5.53351`, `5.40574`, `4.9605`, `4.26749`, `3.49149`, `2.70173`,
`1.9528`, `1.31708`, `0.856883`, `0.561079`, `0.38067`, `0.271014`,
`0.201573`, `0.155615`, `0.124421`, `0.101644`.

## Notes

The first 1024-step attempt used `lr=0.005` and learned the target, but failed
the BF16 export/reference tolerance. The prescribed 256-step sweep found
passing learning runs at `0.001` and `0.003`; `0.005` and `0.01` failed the
same BF16 tolerance. The full accepted rerun used the first passing sweep LR,
`0.001`.

The 40M dense path remains compatibility-only until a longer
`accepted_training` or `dense_learning_control` run passes the same promotion
criteria. Transformer CUDA forward/backward and autoregressive decode remain
future work.

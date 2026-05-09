# Promotion Criteria

Owner: `docs/operations/performance/promotion-criteria.md`.
State: canonical benchmark promotion gates.

## Dense Diagnostics

Compatibility-only 40M start checks remain diagnostic. They must write
`promotion_status=diagnostic_only` and
`run_purpose=bounded_diagnostic_start_check`.

Controlled dense learning runs include the exact Docker train command, train
report capability fields, loss trend evidence, tokens and loss-token counts,
batch, sequence, gradient, checkpoint settings, throughput, timings,
checkpoint, export, logits, inspect, repeated inference checksums, and exact
learning or promotion rejection reasons when the run is not promotable.

## Accepted Dense Training

Accepted-training promotion requires `run_purpose=accepted_training`,
`status=success`, at least 1024 optimizer steps, at least 8 finite loss
samples, `learning_status=learning`, `loss_decrease_fraction >= 0.10`,
last-quarter sampled mean below first-quarter sampled mean, valid `tokens_seen`
and `loss_tokens`, cache row count at least 32, source/tokenizer/config
digests, packed checksum, checkpoint/export/logits checksums, BF16 reference
check pass at tolerance `0.01`, two passing dense inference checks with matching
checksums, positive throughput, and required dense timing/backend metadata.

## Dense Speed Comparison

Dense speed comparisons use matched pre-change and post-change reports. Compare
`timings.backward`, `tokens_per_second`, `config_digest`, `batch_size`,
`seq_len`, `grad_accum`, and `cuda_arch_flags`.

Accept a dense speed slice only when correctness gates pass and backward time
improves or throughput is not more than 5 percent worse.

## Decoder Acceptance

Decoder promotion cannot use `target_seconds > 0` as a shortcut. Accepted
decoder evidence must prove full decoder CUDA training, positive non-embedding
and decoder-block quantitative weight deltas, checkpoint/export/served
artifacts, passing logits checks, supported CUDA KV-cache decode fields, finite
loss, positive optimizer steps, and positive loss-token counts.

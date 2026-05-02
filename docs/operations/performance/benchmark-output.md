# Benchmark Output

Native benchmark tools consume the stable native training report and write
JSON/CSV summaries for aggregate comparison.

## Train Report

Each benchmark repeat copies:

- `DATA_DIR/runs/train-report.json`

The report includes:

- `schema_version`
- `trainer_mode`
- `run_purpose`
- `status`
- `model_kind`
- `accepted_cuda_training`
- `implementation_status`
- `forward_backend`
- `backward_backend`
- `optimizer_backend`
- `cuda_probe_passed`
- `precision_mode`
- `master_dtype`
- `shadow_dtype`
- `accumulation_dtype`
- `export_dtype`
- `cuda_available`
- `cuda_device_name`
- `cuda_arch_flags`
- `git_commit`
- `build_type`
- `config_path`
- `config_digest`
- `dataset_path`
- `dataset_digest`
- transformer shape fields when `model_kind=transformer`: `layers`, `heads`,
  `kv_heads`, `hidden_size`, `head_dim`, `ffn_size`, and `context`
- `parameter_count`
- `optimizer_steps`
- `microsteps`
- `tokens_seen`
- `loss`
- `timings`
- `limitations`
- `capability`
- `checkpoint_checksum`
- `export_checksum`
- `logits_check`

`timings` includes `batch_load`, `h2d`, `forward`, `backward`, `optimizer`,
`checkpoint`, and `export`. `capability` uses the reusable native capability
JSON shape. Dense `logits_check` validates exported BF16 weights and, for train
runs, records FP32 checkpoint reference tolerance fields.

Successful schema v3 train reports use top-level `status=success`. Nested
checks, including `logits_check.status` and `reference_check`, continue to use
`pass` or `fail`.

Dense accepted reports set `accepted_cuda_training=true`. Transformer reports
are retained as experimental records with `accepted_cuda_training=false` and are
excluded from accepted CUDA promotion aggregates.

## Summary JSON

Each summary includes:

- `schema_version`
- `trainer_mode`
- `run_purpose`
- `status`
- `model_kind`
- `accepted_cuda_training`
- `implementation_status`
- `optimizer_steps`
- `microsteps`
- `tokens_seen`
- `initial_loss`
- `loss`
- `median_tokens_per_second`
- `median_step_seconds`
- `mean_h2d_seconds`
- `mean_forward_seconds`
- `mean_backward_seconds`
- `mean_optimizer_seconds`
- `logits_checksum`
- `checkpoint_checksum`
- `export_checksum`
- `logits_check_status`
- `logits_reference_check`
- `logits_max_abs_diff`
- `logits_tolerance`

## Promotion Summary

Dense debug promotions also write
`artifacts/benchmarks/<run-id>/promotion-summary.json`. It records promotion
status, device/backend, batch/sequence/hidden/vocab shape, parameter count,
loss, throughput, elapsed time, H2D and phase timing fractions, artifact
checksums, logits reference-check results, and resume-check results.

Compatibility-only 40M start checks write
`artifacts/benchmarks/<run-id>/dense_40m_compat_4/repeat-01/compatibility-summary.json`
with `promotion_status=compatibility_only` and
`run_purpose=bounded_compatibility_start_check`.

CSV summaries use the same stable names for columns that fit flat tabular
output. Nested capability fields are flattened with a `capability_` prefix.

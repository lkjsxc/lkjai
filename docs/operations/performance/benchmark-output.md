# Benchmark Output

Native benchmark tools consume the stable native training report and write
JSON/CSV summaries for aggregate comparison.

## Train Report

Each benchmark repeat copies:

- `DATA_DIR/runs/train-report.json`

The report includes:

- `schema_version`
- `trainer_mode`
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

`capability` uses the reusable native capability JSON shape. `logits_check`
validates exported BF16 weights.

Dense accepted reports set `accepted_cuda_training=true`. Transformer reports
are retained as experimental records with `accepted_cuda_training=false` and are
excluded from accepted CUDA promotion aggregates.

## Summary JSON

Each summary includes:

- `schema_version`
- `trainer_mode`
- `model_kind`
- `accepted_cuda_training`
- `implementation_status`
- `optimizer_steps`
- `microsteps`
- `tokens_seen`
- `median_tokens_per_second`
- `median_step_seconds`
- `mean_h2d_seconds`
- `mean_forward_seconds`
- `mean_backward_seconds`
- `mean_optimizer_seconds`
- `logits_checksum`
- `checkpoint_checksum`
- `export_checksum`

CSV summaries use the same stable names for columns that fit flat tabular
output. Nested capability fields are flattened with a `capability_` prefix.

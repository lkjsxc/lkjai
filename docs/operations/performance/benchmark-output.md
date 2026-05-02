# Benchmark Output

Native benchmark tools consume the stable native training report and write
JSON/CSV summaries for aggregate comparison.

## Train Report

Each benchmark repeat copies:

- `DATA_DIR/runs/train-report.json`

The report includes:

- `schema_version`
- `trainer_mode`
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
- `optimizer_steps`
- `microsteps`
- `tokens_seen`
- `loss`
- `timings`
- `capability`
- `checkpoint_checksum`
- `export_checksum`
- `logits_check`

`capability` uses the reusable native capability JSON shape. `logits_check`
validates exported BF16 weights.

## Summary JSON

Each summary includes:

- `schema_version`
- `trainer_mode`
- `optimizer_steps`
- `microsteps`
- `tokens_seen`
- `median_tokens_per_second`
- `median_step_seconds`
- `mean_forward_seconds`
- `mean_backward_seconds`
- `mean_optimizer_seconds`
- `logits_checksum`
- `checkpoint_checksum`
- `export_checksum`

CSV summaries use the same stable names for columns that fit flat tabular
output. Nested capability fields are flattened with a `capability_` prefix.

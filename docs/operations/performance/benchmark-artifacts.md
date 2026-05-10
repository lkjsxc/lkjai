# Benchmark Artifacts

Owner: `docs/operations/performance/benchmark-artifacts.md`.
State: canonical generated benchmark artifact shape.

Benchmark tooling consumes `DATA_DIR/runs/train-report.json` or stdout JSON and
writes generated summaries under `artifacts/benchmarks/<run-id>/`.

## Summary JSON

Each summary includes stable flat fields from the train report:

- Identity and acceptance: `schema`, `trainer_mode`, `run_purpose`, `status`,
  `model_kind`, `accepted_cuda_training`, `implementation_status`.
- Progress and shape: `optimizer_steps`, `microsteps`, `tokens_seen`,
  `loss_tokens`, `batch_size`, `seq_len`, `grad_accum`, `parameter_count`.
- Backends and memory: `forward_backend`, `backward_backend`,
  `backward_gemm_enabled`, `embedding_grad_backend`,
  `dense_step_logits_bytes`, `dense_step_grad_logits_bytes`,
  `dense_step_d_hidden_bytes`, `dense_logits_readback_bytes`,
  `cublaslt_workspace_bytes`, `dense_allocator_backend`,
  `dense_workspace_high_water_bytes`, `dense_workspace_reallocations`.
- Timing modes: `dense_timing_mode`, `dense_head_f32_cache_enabled`,
  `dense_head_f32_cache_refreshes`, `loss_readback_mode`,
  `logits_readback_mode`, `dense_stream_count`, `dense_batch_slot_count`,
  `copy_compute_overlap_enabled`, `batch_staging_backend`,
  `optimizer_backend`, `cuda_device_name`.
- Loss and learning: `initial_loss`, `loss`, `loss_samples`,
  `loss_sample_interval`, `best_loss`, `best_loss_step`, `loss_delta`,
  `loss_decrease_fraction`, `first_quarter_loss_mean`,
  `last_quarter_loss_mean`, `learning_status`.
- Aggregate speed: `median_tokens_per_second`, `median_step_seconds`,
  `mean_h2d_seconds`, `mean_forward_seconds`, `mean_backward_seconds`,
  `mean_optimizer_seconds`, `mean_checkpoint_seconds`,
  `mean_export_seconds`.
- Checksums and logits: `logits_checksum`, `checkpoint_checksum`,
  `export_checksum`, `logits_check_status`, `logits_reference_check`,
  `logits_max_abs_diff`, `logits_tolerance`.

## Generated Files

Promoted decoder and dense runs should also publish a reproducible bundle with
`summary.md`, `train-report.json`, `metrics.csv`, loss and throughput plots,
latency plot, `gpu-capability.json`, Nsight Compute and Systems reports,
`config.json`, `tokenizer-digest.txt`, `dataset-manifest.json`, and
`demo-transcript.json`.

- Dense debug promotions write
  `artifacts/benchmarks/<run-id>/promotion-summary.json`.
- Compatibility-only 40M start checks write
  `artifacts/benchmarks/<run-id>/dense_40m_diag_4/repeat-01/diagnostic-summary.json`
  with `promotion_status=diagnostic_only`.
- Controlled dense learning runs write
  `artifacts/benchmarks/<run-id>/dense_learning_control_1024/repeat-01/learning-summary.json`
  and `benchmark-summary.json`.
- Accepted dense training runs write
  `artifacts/benchmarks/<run-id>/dense_accepted_training_1024/repeat-01/accepted-training-summary.json`
  and `benchmark-summary.json`, with matching copies in
  `data/perf-runs/<run-id>/dense_accepted_training_1024/repeat-01/`.

CSV summaries use the same stable names for columns that fit flat tabular
output. Nested capability fields are flattened with a `capability_` prefix.

# Train Report Fields

Owner: `docs/operations/performance/contracts/train-report-fields.md`.
State: canonical field inventory for native train reports.

Each benchmark repeat copies `DATA_DIR/runs/train-report.json`.

## Stable Fields

- Identity: `schema`, `trainer_mode`, `mode`, `run_purpose`, `status`,
  `failure_reason`, `model_kind`, `implementation_status`.
- Acceptance: `accepted_cuda_training`, `dense_cuda_path`,
  `transformer_cuda_path`, `decoder_cuda_path`, `limitations`.
- Backends: `forward_backend`, `backward_backend`, `optimizer_backend`,
  `embedding_grad_backend`, `loss_kernel_backend`, `matmul_backend`,
  `attention_backend`, `decode_backend`, `kv_cache_backend`.
- Precision: `precision_mode`, `master_dtype`, `shadow_dtype`,
  `accumulation_dtype`, `export_dtype`.
- CUDA and build: `cuda_probe_passed`, `cuda_available`, `cuda_device_name`,
  `cuda_arch_flags`, `cuda_driver_version`, `cuda_runtime_version`,
  `cudnn_version`, `cuda_device_count`, `cuda_device_index`,
  `cuda_total_global_memory`, `cuda_sm_count`, `git_commit`, `build_type`.
- Inputs: `config_path`, `config_digest`, `dataset_path`,
  `packed_cache_path`, `dataset_digest`, `train_config_path`, `seed`.
- Shape: `batch_size`, `seq_len`, `grad_accum`, `parameter_count`,
  `target_seconds`, `deadline_hit`, `stop_reason`.
- Transformer and decoder shape fields: `layers`, `heads`, `kv_heads`,
  `hidden_size`, `head_dim`, `ffn_size`, `context`.
- Progress: `optimizer_steps`, `steps`, `start_step`, `microsteps`,
  `tokens_seen`, `input_tokens`, `loss_tokens`.
- Loss: `initial_loss`, `loss`, `loss_finite`, `loss_samples`,
  `loss_sample_interval`, `best_loss`, `best_loss_step`, `loss_delta`,
  `loss_decrease_fraction`, `first_quarter_loss_mean`,
  `last_quarter_loss_mean`, `learning_status`.
- Timings and throughput: `elapsed_ms`, `elapsed_seconds`,
  `tokens_per_second`, `timings`.
- Artifacts: `checkpoint_path`, `checkpoint_checksum`, `export_path`,
  `export_checksum`, `served_path`.
- Logits: `logits_checksum`, `logits_check_passed`, `logits_check`.
- Capability: `capability`.

## Dense Fields

Dense reports also include `backward_gemm_enabled`,
`dense_step_logits_bytes`, `dense_step_grad_logits_bytes`,
`dense_step_d_hidden_bytes`, `dense_logits_readback_bytes`,
`cublaslt_workspace_bytes`, `dense_autotune_enabled`,
`dense_autotune_mode`, `dense_workspace_sweep_bytes`,
`dense_cublaslt_logits_algo_id`, `dense_cublaslt_head_grad_algo_id`,
`dense_cublaslt_hidden_grad_algo_id`,
`dense_cublaslt_logits_workspace_bytes`,
`dense_cublaslt_head_grad_workspace_bytes`,
`dense_cublaslt_hidden_grad_workspace_bytes`, `dense_allocator_backend`,
`dense_async_alloc_supported`, `dense_mempool_release_threshold_bytes`,
`dense_workspace_high_water_bytes`, `dense_workspace_reallocations`,
`dense_timing_mode`, `dense_head_f32_cache_enabled`,
`dense_head_f32_cache_refreshes`, `loss_readback_mode`,
`logits_readback_mode`, `dense_stream_count`, `dense_batch_slot_count`,
`copy_compute_overlap_enabled`, and `batch_staging_backend`.

Dense weight evidence uses `weight_changed` and `weight_change`.

## Decoder Fields

Decoder reports also include `decoder_status`, `embedding_tying`,
`trainable_tensor_count`, `decoder_cuda_slice`, `decoder_block_backend`,
`decoder_block_forward_in_training`, `decoder_block_forward_steps`,
`decoder_forward_probe`, `rmsnorm_backend`, `rope_backend`,
`qkv_projection_backend`, `mlp_backend`, `decoder_backward_backend`,
`workspace_high_water_bytes`, `workspace_reallocations`,
`decode_supported`, `embedding_weight_changed`,
`lm_head_weight_changed`, `non_embedding_weight_changed`,
`decoder_block_weight_changed`, and `decoder_weight_change`.

`decoder_weight_change` records quantitative deltas for embeddings, LM head,
non-embedding tensors, and decoder-block tensors. Partial tied decoder slice
reports must show real embedding and LM-head deltas while keeping
accepted training status false.

Decoder reports must not emit `implementation_status=accepted`,
`decoder_cuda_slice=full_decoder`,
`decoder_backward_backend=cuda_full_decoder`,
`decode_backend=cuda_kv_cache`, or
`kv_cache_backend=cuda_contiguous_bf16` until real full decoder backward and
real CUDA KV-cache decode are implemented and verified. Sidecars such as
`decoder_acceptance.json` belong only to accepted evidence bundles.

# Native Source

## Purpose

Native C++ and CUDA sources implement scratch training, inference serving,
artifact inspection, and CUDA capability probing.

## Contents

- [artifact.cpp](artifact.cpp) and [artifact.hpp](artifact.hpp): native artifact
  read/write helpers.
- [artifact_manifest.cpp](artifact_manifest.cpp) and
  [artifact_manifest.hpp](artifact_manifest.hpp): manifest checksum/schema
  validation.
- [artifact_validate.cpp](artifact_validate.cpp) and
  [artifact_validate.hpp](artifact_validate.hpp): artifact tensor range and
  optimizer index validation.
- [capability_json.cpp](capability_json.cpp) and
  [capability_json.hpp](capability_json.hpp): shared capability JSON fields.
- [cuda_probe.cu](cuda_probe.cu) and [cuda_probe.hpp](cuda_probe.hpp): CUDA
  availability checks.
- [decoder_decode.cpp](decoder_decode.cpp) and
  [decoder_decode.hpp](decoder_decode.hpp): decoder artifact autoregressive
  HTTP chat response helper.
- [decoder_kv_cache.cpp](decoder_kv_cache.cpp),
  [decoder_kv_cache_lifetime.cpp](decoder_kv_cache_lifetime.cpp), and
  [decoder_kv_cache.hpp](decoder_kv_cache.hpp): contiguous BF16 K/V cache
  layout and allocation contract for accepted incremental decoder serving.
- [decoder_cuda_block.cu](decoder_cuda_block.cu),
  [decoder_cuda_block_probe.cpp](decoder_cuda_block_probe.cpp),
  [decoder_cuda_block_project.cu](decoder_cuda_block_project.cu),
  [decoder_cuda_layer_forward.cpp](decoder_cuda_layer_forward.cpp),
  [decoder_cuda_full_forward.cpp](decoder_cuda_full_forward.cpp),
  [decoder_cuda_block.hpp](decoder_cuda_block.hpp), and
  [decoder_cuda_block_internal.hpp](decoder_cuda_block_internal.hpp): stateful
  decoder forward substrate covering RMSNorm, RoPE, BF16 cuBLASLt projections,
  residual adds, down projection, SwiGLU glue, and debug full-forward parity.
- [decoder_cuda_residual.cu](decoder_cuda_residual.cu) and
  [decoder_cuda_residual.hpp](decoder_cuda_residual.hpp): BF16 residual-add
  and residual-backward helpers for decoder probes and later backward wiring.
- [decoder_cuda_slice.cpp](decoder_cuda_slice.cpp),
  [decoder_cuda_state.hpp](decoder_cuda_state.hpp),
  [decoder_cuda_state.cpp](decoder_cuda_state.cpp),
  [decoder_cuda_forward.cpp](decoder_cuda_forward.cpp),
  [decoder_cuda_backward.cpp](decoder_cuda_backward.cpp),
  [decoder_cuda_optimizer.cpp](decoder_cuda_optimizer.cpp),
  [decoder_cuda_slice_internal.hpp](decoder_cuda_slice_internal.hpp), and
  [decoder_cuda_slice_util.cpp](decoder_cuda_slice_util.cpp): persistent
  decoder CUDA state skeleton and partial training report path.
- [dense_cuda.cu](dense_cuda.cu) and [dense_cuda.hpp](dense_cuda.hpp):
  dense CUDA parity and public training/logits entrypoints.
- [dense_demo.cpp](dense_demo.cpp) and [dense_demo.hpp](dense_demo.hpp):
  local dense demo status and next-token JSON helpers.
- [dense_cuda_common.cpp](dense_cuda_common.cpp),
  [dense_cuda_gemm.cu](dense_cuda_gemm.cu),
  [dense_cuda_internal.hpp](dense_cuda_internal.hpp),
  [dense_cuda_kernels.cu](dense_cuda_kernels.cu),
  [dense_cuda_logits.cpp](dense_cuda_logits.cpp),
  [dense_cuda_logits_reference.cpp](dense_cuda_logits_reference.cpp),
  [dense_cuda_report.cpp](dense_cuda_report.cpp),
  [dense_cuda_state.cu](dense_cuda_state.cu),
  [dense_cuda_step.cu](dense_cuda_step.cu),
  [dense_cuda_tuning.cpp](dense_cuda_tuning.cpp),
  [dense_cuda_tuning.hpp](dense_cuda_tuning.hpp), and
  [dense_cuda_train.cpp](dense_cuda_train.cpp): dense BF16 CUDA trainer,
  logits check, cuBLASLt forward/backward GEMM wrapper, scatter-add embedding
  gradient kernel, runtime tuning, report fields, reusable step buffers, and
  state management.
- [dense_checkpoint.cpp](dense_checkpoint.cpp): dense optimizer checkpoint
  restore and matching checks.
- [dense_check_main.cpp](dense_check_main.cpp): dense CUDA check binary.
- [dense_loss_trend.cpp](dense_loss_trend.cpp) and
  [dense_loss_trend.hpp](dense_loss_trend.hpp): dense sampled-loss trend
  classification helpers.
- [dense_report_util.cpp](dense_report_util.cpp) and
  [dense_report_util.hpp](dense_report_util.hpp): dense report checksum,
  parameter-count, and loss-sample serialization helpers.
- [dense_weight_change.cpp](dense_weight_change.cpp) and
  [dense_weight_change.hpp](dense_weight_change.hpp): dense tensor delta
  summaries used by accepted training reports.
- [runtime_context.cu](runtime_context.cu), [runtime_device.cu](runtime_device.cu),
  [runtime_workspace.cu](runtime_workspace.cu), [runtime_errors.cu](runtime_errors.cu),
  and [runtime_device.hpp](runtime_device.hpp):
  reusable CUDA tensor/context substrate for later GEMM and attention work.
- [runtime_device_check_main.cpp](runtime_device_check_main.cpp): BF16
  device-tensor round-trip check binary.
- [dense_model.cpp](dense_model.cpp) and [dense_model.hpp](dense_model.hpp):
  legacy artifact helpers kept for old inspect fixtures.
- [dense_train.cpp](dense_train.cpp), [dense_train_artifact.cpp](dense_train_artifact.cpp),
  [dense_train_math.cpp](dense_train_math.cpp), [dense_train.hpp](dense_train.hpp),
  and [dense_train_internal.hpp](dense_train_internal.hpp): dense config,
  artifact, and CPU reference math used by the active CUDA trainer.
- [env.cpp](env.cpp) and [env.hpp](env.hpp): environment parsing helpers.
- [http_server.cpp](http_server.cpp) and [http_server.hpp](http_server.hpp):
  minimal HTTP server.
- [inspect_main.cpp](inspect_main.cpp): artifact inspection binary.
- [infer_main.cpp](infer_main.cpp): dense BF16 export logits inference binary.
- [json_min.cpp](json_min.cpp) and [json_min.hpp](json_min.hpp): small JSON
  helpers.
- [logits_check_main.cpp](logits_check_main.cpp): dense and transformer artifact
  logits probe, including dense BF16 export parity against FP32 checkpoints.
- [native_tokenizer_build.cpp](native_tokenizer_build.cpp) and
  [native_tokenizer_build.hpp](native_tokenizer_build.hpp): deterministic
  native byte-level BPE-compatible tokenizer builder with atomic decoder tags.
- [packed_cache.cpp](packed_cache.cpp) and [packed_cache.hpp](packed_cache.hpp):
  packed-cache validation.
- [packed_cache_build.cpp](packed_cache_build.cpp) and
  [packed_cache_build.hpp](packed_cache_build.hpp): native packed-cache build
  and validate CLI support.
- [packed_cache_digest.cpp](packed_cache_digest.cpp) and
  [packed_cache_digest.hpp](packed_cache_digest.hpp): packed-cache checksum and
  digest helpers.
- [packed_cache_reader.cpp](packed_cache_reader.cpp): persistent packed-cache
  reader for dense training batches.
- [packed_cache_validate.cpp](packed_cache_validate.cpp): packed-cache metadata,
  file size, and row-bound validation.
- [packed_cache_main.cpp](packed_cache_main.cpp): packed-cache build and
  validation CLI.
- [server_main.cpp](server_main.cpp): merged native server entrypoint for
  OpenAI-compatible inference routes and local runtime API routes.
- [native_server_routes.cpp](native_server_routes.cpp) and
  [native_server_routes.hpp](native_server_routes.hpp): testable merged route
  dispatcher for `/`, `/healthz`, `/api/*`, and `/v1/*`.
- [runtime_main.cpp](runtime_main.cpp), [runtime_api.cpp](runtime_api.cpp),
  [runtime_api.hpp](runtime_api.hpp), [runtime_action.cpp](runtime_action.cpp),
  [runtime_action.hpp](runtime_action.hpp), [runtime_agent.cpp](runtime_agent.cpp),
  [runtime_agent.hpp](runtime_agent.hpp), [runtime_events.cpp](runtime_events.cpp),
  [runtime_events.hpp](runtime_events.hpp),
  [native_http_client.cpp](native_http_client.cpp), and
  [native_http_client.hpp](native_http_client.hpp): native agent API runtime,
  XML action loop, transcript/event contracts, and blocking HTTP model client.
- [repo_check_main.cpp](repo_check_main.cpp) and `repo_check_*`: native docs,
  corpus, and repository quality checks.
- [train_main.cpp](train_main.cpp): scratch training entrypoint.
- [train_data.cpp](train_data.cpp) and [train_data.hpp](train_data.hpp):
  JSONL corpus cursor and row extraction helpers.
- [train_report_digest.cpp](train_report_digest.cpp) and
  [train_report_digest.hpp](train_report_digest.hpp): shared report digest and
  artifact checksum helpers.
- [train_real.cpp](train_real.cpp) and [train_real.hpp](train_real.hpp):
  corpus-backed native training loop used by non-smoke runs.
- [train_report.cpp](train_report.cpp), [train_report.hpp](train_report.hpp),
  [transformer_report.cpp](transformer_report.cpp), and
  [transformer_report_io.cpp](transformer_report_io.cpp): train-report
  JSON writers.
- [transformer_report_acceptance.cpp](transformer_report_acceptance.cpp) and
  [transformer_report_acceptance.hpp](transformer_report_acceptance.hpp):
  decoder acceptance and limitation helpers for report contracts.
- [training_config.cpp](training_config.cpp) and
  [training_config.hpp](training_config.hpp): JSON training-run config loading
  and CLI/environment precedence.
- [transformer_artifact.cpp](transformer_artifact.cpp),
  [transformer_config.cpp](transformer_config.cpp),
  [transformer_cuda.cu](transformer_cuda.cu),
  [transformer_forward.cpp](transformer_forward.cpp),
  [transformer_init.cpp](transformer_init.cpp),
  [transformer_load.cpp](transformer_load.cpp),
  [transformer_logits.cpp](transformer_logits.cpp),
  [transformer_optim.cpp](transformer_optim.cpp),
  [transformer_state.hpp](transformer_state.hpp),
  [transformer_train.cpp](transformer_train.cpp),
  [transformer_train.hpp](transformer_train.hpp), and
  [transformer_util.cpp](transformer_util.cpp): experimental transformer
  training, artifact, checkpoint, report, and logits-check implementation.

## Rules

- Keep native source files at `<= 200` lines.
- Build and test through Docker Compose.
- Dense CUDA optimization ownership lives in the `dense_cuda_*` files. Keep
  transformer CUDA, decode, graphs, and NCCL changes in their own documented
  phases.

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
- [dense_cuda.cu](dense_cuda.cu) and [dense_cuda.hpp](dense_cuda.hpp):
  dense CUDA parity and public training/logits entrypoints.
- [dense_cuda_common.cpp](dense_cuda_common.cpp),
  [dense_cuda_gemm.cu](dense_cuda_gemm.cu),
  [dense_cuda_internal.hpp](dense_cuda_internal.hpp),
  [dense_cuda_kernels.cu](dense_cuda_kernels.cu),
  [dense_cuda_logits.cpp](dense_cuda_logits.cpp),
  [dense_cuda_logits_reference.cpp](dense_cuda_logits_reference.cpp),
  [dense_cuda_state.cu](dense_cuda_state.cu), and
  [dense_cuda_train.cpp](dense_cuda_train.cpp): dense BF16 CUDA trainer,
  logits check, GEMM wrapper, kernels, and state management.
- [dense_checkpoint.cpp](dense_checkpoint.cpp): dense optimizer checkpoint
  restore and compatibility checks.
- [dense_check_main.cpp](dense_check_main.cpp): dense CUDA check binary.
- [runtime_device.cu](runtime_device.cu), [runtime_workspace.cu](runtime_workspace.cu),
  [runtime_errors.cu](runtime_errors.cu), and [runtime_device.hpp](runtime_device.hpp):
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
- [json_min.cpp](json_min.cpp) and [json_min.hpp](json_min.hpp): small JSON
  helpers.
- [logits_check_main.cpp](logits_check_main.cpp): dense and transformer artifact
  logits probe, including dense BF16 export parity against FP32 checkpoints.
- [packed_cache.cpp](packed_cache.cpp) and [packed_cache.hpp](packed_cache.hpp):
  packed-cache v2 validation and compatible v1 migration.
- [packed_cache_validate.cpp](packed_cache_validate.cpp): packed-cache metadata,
  file size, and row-bound validation.
- [packed_cache_main.cpp](packed_cache_main.cpp): packed-cache migration CLI.
- [server_main.cpp](server_main.cpp): OpenAI-compatible inference entrypoint.
- [train_main.cpp](train_main.cpp): scratch training entrypoint.
- [train_data.cpp](train_data.cpp) and [train_data.hpp](train_data.hpp):
  JSONL corpus cursor and row extraction helpers.
- [train_real.cpp](train_real.cpp) and [train_real.hpp](train_real.hpp):
  corpus-backed native training loop used by non-smoke runs.
- [train_report.cpp](train_report.cpp), [train_report.hpp](train_report.hpp),
  and [transformer_report.cpp](transformer_report.cpp): stable train-report
  JSON writers.
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

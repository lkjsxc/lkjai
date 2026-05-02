# Native Source

## Purpose

Native C++ and CUDA sources implement scratch training, inference serving,
artifact inspection, and CUDA capability probing.

## Contents

- [artifact.cpp](artifact.cpp) and [artifact.hpp](artifact.hpp): native artifact
  read/write helpers.
- [cuda_probe.cu](cuda_probe.cu) and [cuda_probe.hpp](cuda_probe.hpp): CUDA
  availability checks.
- [dense_cuda.cu](dense_cuda.cu) and [dense_cuda.hpp](dense_cuda.hpp):
  BF16/cuBLASLt/cuDNN smoke checks.
- [dense_check_main.cpp](dense_check_main.cpp): dense CUDA check binary.
- [dense_model.cpp](dense_model.cpp) and [dense_model.hpp](dense_model.hpp):
  legacy artifact helpers kept for old inspect fixtures.
- [dense_train.cpp](dense_train.cpp), [dense_train_artifact.cpp](dense_train_artifact.cpp),
  [dense_train_math.cpp](dense_train_math.cpp), [dense_train.hpp](dense_train.hpp),
  and [dense_train_internal.hpp](dense_train_internal.hpp): legacy packed-cache
  trainer substrate retained outside the active training entrypoint.
- [env.cpp](env.cpp) and [env.hpp](env.hpp): environment parsing helpers.
- [http_server.cpp](http_server.cpp) and [http_server.hpp](http_server.hpp):
  minimal HTTP server.
- [inspect_main.cpp](inspect_main.cpp): artifact inspection binary.
- [json_min.cpp](json_min.cpp) and [json_min.hpp](json_min.hpp): small JSON
  helpers.
- [logits_check_main.cpp](logits_check_main.cpp): transformer artifact logits
  probe.
- [packed_cache.cpp](packed_cache.cpp) and [packed_cache.hpp](packed_cache.hpp):
  packed-cache v2 validation and compatible v1 migration.
- [packed_cache_main.cpp](packed_cache_main.cpp): packed-cache migration CLI.
- [server_main.cpp](server_main.cpp): OpenAI-compatible inference entrypoint.
- [train_main.cpp](train_main.cpp): scratch training entrypoint.
- [train_data.cpp](train_data.cpp) and [train_data.hpp](train_data.hpp):
  JSONL corpus cursor and row extraction helpers.
- [train_real.cpp](train_real.cpp) and [train_real.hpp](train_real.hpp):
  corpus-backed native training loop used by non-smoke runs.
- [transformer_artifact.cpp](transformer_artifact.cpp), [transformer_config.cpp](transformer_config.cpp),
  [transformer_forward.cpp](transformer_forward.cpp), [transformer_init.cpp](transformer_init.cpp),
  [transformer_optim.cpp](transformer_optim.cpp), [transformer_state.hpp](transformer_state.hpp),
  [transformer_train.cpp](transformer_train.cpp), and [transformer_train.hpp](transformer_train.hpp):
  active packed-cache BF16 transformer train/export/logits implementation.

## Rules

- Keep native source files at `<= 200` lines.
- Build and test through Docker Compose.

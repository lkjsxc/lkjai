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
  dense smoke artifact and decode helpers.
- [env.cpp](env.cpp) and [env.hpp](env.hpp): environment parsing helpers.
- [http_server.cpp](http_server.cpp) and [http_server.hpp](http_server.hpp):
  minimal HTTP server.
- [inspect_main.cpp](inspect_main.cpp): artifact inspection binary.
- [json_min.cpp](json_min.cpp) and [json_min.hpp](json_min.hpp): small JSON
  helpers.
- [server_main.cpp](server_main.cpp): OpenAI-compatible inference entrypoint.
- [train_main.cpp](train_main.cpp): scratch training entrypoint.
- [train_data.cpp](train_data.cpp) and [train_data.hpp](train_data.hpp):
  JSONL corpus cursor and row extraction helpers.
- [train_real.cpp](train_real.cpp) and [train_real.hpp](train_real.hpp):
  corpus-backed native training loop used by non-smoke runs.

## Rules

- Keep native source files at `<= 200` lines.
- Build and test through Docker Compose.

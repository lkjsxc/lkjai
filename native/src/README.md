# Native Source

## Purpose

Native C++ and CUDA sources implement scratch training, inference serving,
artifact inspection, and CUDA capability probing.

## Contents

- [artifact.cpp](artifact.cpp) and [artifact.hpp](artifact.hpp): native artifact
  read/write helpers.
- [cuda_probe.cu](cuda_probe.cu), [cuda_probe.hpp](cuda_probe.hpp), and
  [cuda_probe_stub.cpp](cuda_probe_stub.cpp): CUDA availability checks.
- [env.cpp](env.cpp) and [env.hpp](env.hpp): environment parsing helpers.
- [http_server.cpp](http_server.cpp) and [http_server.hpp](http_server.hpp):
  minimal HTTP server.
- [inspect_main.cpp](inspect_main.cpp): artifact inspection binary.
- [json_min.cpp](json_min.cpp) and [json_min.hpp](json_min.hpp): small JSON
  helpers.
- [server_main.cpp](server_main.cpp): OpenAI-compatible inference entrypoint.
- [simple_model.cpp](simple_model.cpp) and [simple_model.hpp](simple_model.hpp):
  compact model implementation.
- [train_main.cpp](train_main.cpp): scratch training entrypoint.

## Rules

- Keep native source files at `<= 200` lines.
- Build and test through Docker Compose.

# Native Product Runtime

Owner: `native/README.md`.
State: canonical documentation.


This directory owns the C++/CUDA product path.

## Contents

- [CMakeLists.txt](CMakeLists.txt): native build graph.
- [cmake/](cmake/README.md): CMake source-list fragments.
- [src/](src/): server, trainer, artifact, and CUDA probe code.
- [tests/](tests/README.md): native CTest helper scripts.

## Build

Use Docker Compose from the repository root. Host-local CUDA tooling is not
required.

# Native Product Runtime

Owner: `docs/architecture/native/README.md`.
State: navigation and local operator notes.


This directory contains the C++/CUDA product path. Behavior contracts live
under `docs/`; keep this file limited to navigation and local build notes.

## Contents

- [CMakeLists.txt](CMakeLists.txt): native build graph.
- [cmake/](cmake/README.md): CMake source-list fragments.
- [src/](src/): server, trainer, artifact, and CUDA probe code.
- [tests/](tests/README.md): native CTest helper scripts.

## Build

Use Docker Compose from the repository root. Host-local CUDA tooling is not
required.

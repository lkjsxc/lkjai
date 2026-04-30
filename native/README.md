# Native Product Runtime

This directory owns the C++/CUDA product path.

## Contents

- [CMakeLists.txt](CMakeLists.txt): native build graph.
- [src/](src/): server, trainer, artifact, and CUDA probe code.

## Build

Use Docker Compose from the repository root. Host-local CUDA tooling is not
required.

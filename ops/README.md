# Ops

Owner: `ops/README.md`.
State: canonical documentation.


## Purpose

Operational files for building, running, and verifying the project live here.

## Contents

- [corpus/](corpus/): isolated public corpus acquisition and validation entry
  points.
- [docker/](docker/): Dockerfiles for runtime, inference, training, and verify
  containers.
- [host/](host/): host setup helpers for NVIDIA container runtime support.
- [capture-decoder-route-transcript.sh](capture-decoder-route-transcript.sh):
  probes the decoder route and writes the accepted route transcript artifact.
- [verify.sh](verify.sh): mandatory verification script used by Compose.
- [verify-cuda-decoder.sh](verify-cuda-decoder.sh): focused CUDA decoder
  verification helper.

## Rules

- Keep Compose at the repository root as `compose.yaml`.
- Run verification through Docker Compose.

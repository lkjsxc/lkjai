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
- [verify.sh](verify.sh): mandatory verification script used by Compose.

## Rules

- Keep Compose at the repository root as `compose.yaml`.
- Run verification through Docker Compose.

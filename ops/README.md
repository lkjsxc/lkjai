# Ops

## Purpose

Operational files for building, running, and verifying the project live here.

## Contents

- [docker/](docker/): Dockerfiles for runtime, inference, training, and verify
  containers.
- [verify.sh](verify.sh): mandatory verification script used by Compose.

## Rules

- Keep Compose at the repository root as `compose.yaml`.
- Run verification through Docker Compose.

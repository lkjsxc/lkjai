# Docker

Owner: `ops/docker/README.md`.
State: canonical documentation.


## Purpose

Dockerfiles define reproducible containers for each execution mode.

## Contents

- [Dockerfile.native](Dockerfile.native): native inference, runtime, train, and
  utility image.
- [Dockerfile.corpus](Dockerfile.corpus): isolated public corpus acquisition
  image.
- [Dockerfile.verify](Dockerfile.verify): combined verification image.

## Rules

- Keep build contexts rooted at the repository root.
- Use paths that match `compose.yaml`.

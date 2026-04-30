# Docker

## Purpose

Dockerfiles define reproducible containers for each execution mode.

## Contents

- [Dockerfile.runtime](Dockerfile.runtime): Rust runtime service image.
- [Dockerfile.native](Dockerfile.native): native train and inference image.
- [Dockerfile.verify](Dockerfile.verify): combined verification image.

## Rules

- Keep build contexts rooted at the repository root.
- Use paths that match `compose.yaml`.

# Corpus Ops

## Purpose

This directory contains isolated public corpus acquisition and validation
entrypoints used only by the `corpus` Compose profile.

## Contents

- [entrypoint.sh](entrypoint.sh): profile command dispatcher.
- [public_pretrain_tool](public_pretrain_tool): corpus-only acquisition,
  preparation, packed-cache, and validation helper.

## Rules

- Keep this tooling outside the product train, serve, runtime, and verification
  paths.
- Prefer native product tooling when a capability graduates out of corpus
  acquisition.
- Split this directory's helper files before adding more behavior.

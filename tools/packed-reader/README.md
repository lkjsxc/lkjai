# Packed Reader

## Purpose

This Rust utility exercises packed-token dataset reading outside the runtime
service.

## Contents

- [src/](src/): utility source code.
- [Cargo.toml](Cargo.toml): crate manifest for package `lkjai_packed_reader`.

## Rules

- Build through the root Cargo workspace.
- Keep utility behavior independent from runtime web serving.

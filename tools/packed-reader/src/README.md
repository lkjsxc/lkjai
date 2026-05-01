# Packed Reader Source

## Purpose

This source directory contains the standalone packed-cache reader utility.

## Contents

- [main.rs](main.rs): CLI entrypoint for reading packed token cache metadata and
  sample records.

## Rules

- Keep the utility independent from the runtime web service.
- Build through the root Cargo workspace.

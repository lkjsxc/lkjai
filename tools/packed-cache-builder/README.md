# Packed Cache Builder

This Rust utility builds deterministic `lkjai-packed-cache-v2` token caches from
JSONL source rows using a HuggingFace `tokenizer.json`.

It is intentionally separate from native training code so packed-cache
construction uses the canonical tokenizer implementation instead of duplicating
tokenization in C++.

## Contents

- [src/](src/): builder and validator source code.
- [Cargo.toml](Cargo.toml): crate manifest for package
  `lkjai_packed_cache_builder`.

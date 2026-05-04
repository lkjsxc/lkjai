# Native Tests

## Purpose

This directory contains test helpers for native CTest cases.

## Contents

- [packed_cache_reader_check.cpp](packed_cache_reader_check.cpp): persistent
  packed-cache reader, wraparound, mismatch, corrupt starts, and truncated file
  rejection.
- [decoder_tokenizer_contract.cpp](decoder_tokenizer_contract.cpp): atomic
  XML-like tags, ordered prompt serialization, and tokenizer round trip.
- [decoder_cuda_norm_check.cpp](decoder_cuda_norm_check.cpp): BF16 CUDA
  RMSNorm parity against a CPU reference.

Repository, docs, corpus, and line-limit checks now live in
`lkjai-native-repo-check` under `native/src/`.

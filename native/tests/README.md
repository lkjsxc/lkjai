# Native Tests

## Purpose

This directory contains test helpers for native CTest cases.

## Contents

- [server_chat_contract.py](server_chat_contract.py): starts the native server
  against a dense export and checks chat returns explicit unsupported decode.
- [packed_train_contract.py](packed_train_contract.py): creates a minimal
  packed-cache v2 dataset, checks dense CUDA training, artifact schema, logits,
  BF16 export parity against FP32 checkpoint masters, true checkpoint resume,
  and deterministic continuation.
- [packed_cache_migration_contract.py](packed_cache_migration_contract.py):
  checks v1-to-v2 cache migration and trains from the migrated cache.
- [benchmark_report_parser.py](benchmark_report_parser.py): checks benchmark
  parsing of the stable native train-report JSON contract.

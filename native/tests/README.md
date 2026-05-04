# Native Tests

## Purpose

This directory contains test helpers for native CTest cases.

## Contents

- [config_contract.py](config_contract.py): validates native and training config
  keys, BF16 dtype, dimension invariants, vocab/context bounds, and local
  config references.
- [cuda_arch_contract.py](cuda_arch_contract.py): validates the CMake, Docker,
  and Compose CUDA architecture policy, including Blackwell `120` defaults.
- [dense_check_contract.py](dense_check_contract.py): wraps the dense CUDA
  parity binary and checks additive hardware/build capability fields.
- [server_chat_contract.py](server_chat_contract.py): starts the native server
  against a dense export and checks chat returns explicit unsupported decode.
- [packed_train_contract.py](packed_train_contract.py): creates a minimal
  packed-cache v2 dataset, checks dense CUDA training, artifact schema, logits,
  BF16 export parity against FP32 checkpoint masters, true checkpoint resume,
  deterministic continuation, and dense backward GEMM/scatter report fields.
- [packed_cache_reader_check.cpp](packed_cache_reader_check.cpp): checks the
  persistent packed-cache reader, wraparound, mismatch, corrupt starts, and
  truncated file rejection.
- [dense_infer_contract.py](dense_infer_contract.py): checks dense BF16 export
  logits inference JSON and invalid artifact/token rejection.
- [packed_cache_migration_contract.py](packed_cache_migration_contract.py):
  checks v1-to-v2 cache migration and trains from the migrated cache.
- [transformer_train_contract.py](transformer_train_contract.py): checks the
  experimental transformer training contract and report status.
- [decoder_train_contract.py](decoder_train_contract.py): checks decoder
  train/export/logits and server chat `choices` contract.
- [decoder_acceptance_gate.py](decoder_acceptance_gate.py): checks that P0 and
  partial CUDA decoder reports cannot pass full acceptance.
- [benchmark_report_parser.py](benchmark_report_parser.py): checks benchmark
  parsing of the stable native train-report JSON contract, including
  compatibility-only run-purpose filtering and dense step-buffer fields.
- [accepted_training_report_validation.py](accepted_training_report_validation.py):
  CTest-wired check for accepted dense training summary validation and token
  accounting.
- [promote_dense_debug_validation.py](promote_dense_debug_validation.py):
  dry-checks dense debug promotion validation and summary shaping.

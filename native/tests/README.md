# Native Tests

## Purpose

This directory contains test helpers for native CTest cases.

## Contents

- [server_chat_contract.py](server_chat_contract.py): starts the native server
  against a dense export and checks chat returns explicit unsupported decode.
- [packed_train_contract.py](packed_train_contract.py): creates a minimal
  packed-cache v2 dataset, checks dense CUDA training, artifact schema, logits,
  cache migration, and checkpoint resume step numbering.

# Native Tests

## Purpose

This directory contains test helpers for native CTest cases.

## Contents

- [server_chat_contract.py](server_chat_contract.py): starts the native server
  against a smoke artifact and checks chat completion behavior.
- [packed_train_contract.py](packed_train_contract.py): creates a minimal
  packed-cache v2 dataset and checks `lkjai-native-train --train`.

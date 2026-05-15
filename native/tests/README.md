# Native Tests

Owner: `native/tests/README.md`.
State: canonical documentation.


## Purpose

This directory contains test helpers for native CTest cases.

## Contents

- [packed_cache_reader_check.cpp](packed_cache_reader_check.cpp): persistent
  packed-cache reader, wraparound, mismatch, corrupt starts, and truncated file
  rejection.
- [packed_cache_strict_check.cpp](packed_cache_strict_check.cpp): strict
  packed-cache build and validation metadata rejection checks.
- [decoder_tokenizer_contract.cpp](decoder_tokenizer_contract.cpp): atomic
  XML-like tags, ordered prompt serialization, and tokenizer round trip.
- [decoder_kv_cache_contract.cpp](decoder_kv_cache_contract.cpp): contiguous
  BF16 K/V cache offset and size contract for the tied 40M decoder.
- [decoder_cuda_norm_check.cpp](decoder_cuda_norm_check.cpp): BF16 CUDA
  RMSNorm forward and backward parity against a CPU reference.
- [decoder_cuda_rope_backward_check.cpp](decoder_cuda_rope_backward_check.cpp):
  BF16 RoPE inverse-gradient parity against a CPU reference.
- [../src/decoder_cuda_block_check.cpp](../src/decoder_cuda_block_check.cpp):
  decoder forward-substrate CTest for RMSNorm, RoPE, projection metadata,
  finite BF16 projection outputs, and truthful partial report fields.
- [runtime_contract_check.cpp](runtime_contract_check.cpp): native runtime
  event filtering, transcript persistence, and model-status JSON contract.
- [runtime_action_contract.cpp](runtime_action_contract.cpp): XML action parser
  validation and repeat-action signature checks.
- [runtime_agent_contract.cpp](runtime_agent_contract.cpp): `agent.finish`,
  `agent.think`, repeat-action, and unsupported-tool loop behavior.
- [server_route_contract.cpp](server_route_contract.cpp): inference route
  contract for model, health, CORS preflight, and rejected sandbox/frontend
  routes.
- [server_route_dense_contract.cpp](server_route_dense_contract.cpp): inference
  rejection contract for removed dense API routes.
- [decoder_acceptance_report_contract.cpp](decoder_acceptance_report_contract.cpp):
  decoder accepted-report guard for full CUDA training and KV-cache decode.

Repository, docs, corpus, and line-limit checks now live in
`lkjai-native-repo-check` under `native/src/`.

# Decoder Acceptance Matrix

Owner: `docs/architecture/native/decoder/acceptance.md`.
State: canonical decoder acceptance matrix.

## States

| State | Report Fields | Meaning |
|---|---|---|
| CUDA forward substrate | `accepted_cuda_training=false`, `forward_backend=cuda_full_decoder`, `attention_backend=cuda_causal_gqa_bf16_reference` | Full decoder forward, loss, logits capture, registry tensors, BF16 shadows, and CUDA AdamW substrate execute. |
| Diagnostic backward | `decoder_backward_backend=cuda_diagnostic_synthetic`, `decoder_gradient_source=cuda_device_diagnostic` | CUDA-resident helper gradients exercise registry updates. This is not chain-rule decoder backward and is not promotable. |
| Accepted training | `implementation_status=accepted`, `accepted_cuda_training=true`, `attention_backend=cudnn_sdpa_bf16_gqa`, `decoder_backward_backend=cuda_full_decoder`, `decoder_gradient_source=cuda_device` | Full decoder forward and reverse-mode backward execute from CUDA tape using accepted cuDNN SDPA attention and device gradients. |
| Accepted decode | `decode_backend=cuda_kv_cache`, `kv_cache_backend=cuda_contiguous_bf16` | The served artifact executes CUDA KV-cache prefill and allocation-free steady decode after accepted training evidence. |

## Accepted Report Requirements

Accepted decoder reports must include all of these fields and evidence:

- `model_kind=decoder`
- `implementation_status=accepted`
- `accepted_cuda_training=true`
- `decoder_cuda_path=true`
- `decoder_cuda_slice=full_decoder`
- `forward_backend=cuda_full_decoder`
- `backward_backend=cuda_full_decoder`
- `decoder_backward_backend=cuda_full_decoder`
- `decoder_gradient_source=cuda_device`
- `attention_backend=cudnn_sdpa_bf16_gqa`
- `optimizer_backend=cuda_adamw_fp32_registry`
- `decode_supported=true`
- `decode_backend=cuda_kv_cache`
- `kv_cache_backend=cuda_contiguous_bf16`
- positive `kv_cache_prefill_allocated_bytes`
- zero `kv_cache_steady_state_token_allocations`
- `logits_check_passed=true`
- finite loss, positive optimizer steps, and positive loss tokens
- positive non-embedding and decoder-block weight deltas
- tied `tok_embeddings:lm_head`
- 40M RTX 3070 shape and two-hour training config
- checkpoint, export, served artifact, and route transcript evidence

## Rejected Promotion Sources

The acceptance guards reject these as accepted evidence:

- `cuda_diagnostic_synthetic`
- `cuda_device_diagnostic`
- `host_reference` gradients or backward
- `cuda_causal_gqa_bf16_reference` as accepted attention
- partial decode names such as `cuda_reference_kv_cache`
- accepted sidecars without a matching accepted train report
- accepted decode fields without executed KV allocation counters

## Required Final Gate

The repository verification gate remains:

```bash
docker compose --progress quiet --profile verify run --build --rm verify
```

The final promotion gate is the real two-hour RTX 3070 run from
`configs/training/decoder_2h_40m_3070.json`, followed by artifact inspection,
logits check, served route execution, and transcript capture.

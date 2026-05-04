# Decoder Config

## Model Kind

- `model_kind`: `decoder`
- Artifact manifest `kind`: `decoder`
- Training objective: `causal_lm_full` first, followed by
  `assistant_masked_sft` when that native objective is implemented.
- Serving target: OpenAI-compatible `/v1/chat/completions`.

## Architecture

The first accepted decoder is a pre-norm dense decoder-only transformer:

- RMSNorm before attention and MLP.
- RoPE on Q/K.
- Grouped-query causal self-attention.
- SwiGLU MLP.
- Untied token embeddings and LM head for the first accepted path.
- BF16 serving weights with FP32 accumulation where numerically required.

## Presets

| File | Role | Shape |
|---|---|---|
| `decoder_18m_bf16_3070.json` | first same-model demo | 8 layers, hidden 512, 8 heads, 2 KV heads, FFN 1536, seq 1024 |
| `decoder_40m_bf16_3070.json` | 3070 profile target | 10 layers, hidden 576, 8 heads, 2 KV heads, FFN 1536, seq 1024 |
| `decoder_140m_bf16_5090.json` | Blackwell profile target | 20 layers, hidden 768, 12 heads, 4 KV heads, FFN 3072, seq 2048 |

RTX 3070 remains the acceptance gate. RTX 5090 results are profile evidence
until the 3070 gate also passes.

## Config Validation

- `dtype` must be `bf16`.
- `heads * head_dim == hidden_size`.
- `heads` must be divisible by `kv_heads`.
- `head_dim` must be a multiple of `8`.
- `activation` must be `swiglu`.
- `context` must match the packed-cache sequence length for training.

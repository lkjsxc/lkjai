# Decoder Decode

## API

`POST /v1/chat/completions` returns OpenAI-compatible JSON for decoder
artifacts:

- `id`
- `object`
- `model`
- `choices[0].index`
- `choices[0].message.role=assistant`
- `choices[0].message.content`
- `choices[0].finish_reason`
- `choices[0].lkjai_stop_reason`
- `choices[0].lkjai_decode_backend`
- `choices[0].lkjai_kv_cache_backend`
- `usage.prompt_tokens`
- `usage.completion_tokens`
- `usage.total_tokens`

Dense and transformer artifacts continue to return HTTP `422` unsupported
decode with no `choices`.

Current decoder artifacts return `lkjai_decode_backend=host_reference_recompute`
and `lkjai_kv_cache_backend=none`. Those fields are deliberately visible so
partial decoder serving is not confused with accepted KV-cache decode.
Training reports mirror this with `decode_backend=host_reference_recompute` and
`kv_cache_backend=none`; the new CUDA forward-substrate probe does not change
decode behavior.

Host-reference decode recomputes the full prompt each token. It uses decoder
token embeddings, RMSNorm, RoPE on Q/K, causal GQA attention, SwiGLU, final
norm, and LM head. It must not add learned `pos_embeddings`.

## Prompt And Tokenizer

The server loads artifact `tokenizer.json`, parses ordered OpenAI-style
`messages[]`, serializes them into the documented XML-like prompt format,
encodes prompt tokens, and decodes generated tokens back to assistant text.

The native tokenizer bridge implements the repo byte-level BPE subset needed by
the local tokenizer: added atomic tags, byte-level prompt text, BPE merges, and
special-token skipping on decode. Prompt serialization preserves raw message
content for training compatibility, uses paired XML-like tags, and ends with
`<assistant_action>\n`.

Serialization order is deterministic:

```text
<dialogue>
<message>
<role>ROLE</role>
<tool_name>NAME</tool_name>
<content>CONTENT</content>
</message>
</dialogue>
<assistant_action>
```

`<tool_name>` is emitted only when a tool name is present. Roles must be one of
`system`, `user`, `assistant`, or `tool`; malformed decoder chat requests return
HTTP `400` with no `choices`.

## Decode Loop

- Prefill consumes the prompt up to the configured context.
- Incremental decode appends one token at a time.
- Accepted decode uses a native-owned contiguous BF16 KV cache.
- The current decoder bridge recomputes the host/reference forward path and
  reports `lkjai_kv_cache_backend=none`.
- Paged KV cache is a later batching optimization.
- Steady-state accepted decode must not allocate device memory per token.

## Sampling

Supported request controls:

- `max_tokens`
- `temperature`
- `top_k`
- `top_p`
- `seed`

Stop when the model emits `<eos>`, `</action>`, or the `max_tokens` cap.
`finish_reason` is `stop` for `<eos>` or `</action>` and `length` for the token
cap. The additive `lkjai_stop_reason` is `eos`, `end_action`, or `max_tokens`.

Sampler validation rejects invalid values instead of clamping:

- `max_tokens` must be in `[1,512]`.
- `temperature` must be finite and non-negative.
- `top_k` must be in `[0,vocab_size]`.
- `top_p` must be in `(0,1]`.
- `seed` must be non-negative.

`temperature=0` uses deterministic argmax.

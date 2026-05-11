# Decoder Decode

Owner: `docs/architecture/native/decoder/decode.md`.
State: accepted when `decode_backend=cuda_kv_cache`.

## API

`POST /v1/chat/completions` may return OpenAI-compatible JSON for decoder
artifacts. Current decoder choices are partial host-reference usability until
accepted CUDA KV-cache decode exists:

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
- `choices[0].lkjai_kv_prefill_allocated_bytes`
- `choices[0].lkjai_kv_steady_state_token_allocations`
- `choices[0].lkjai_decode_supported`
- `usage.prompt_tokens`
- `usage.completion_tokens`
- `usage.total_tokens`

Dense and transformer artifacts continue to return HTTP `422` unsupported
decode with no `choices`.

Accepted decoder artifacts return `lkjai_decode_backend=cuda_kv_cache` and
`lkjai_kv_cache_backend=cuda_contiguous_bf16`. Reports and route responses also
disclose positive prefill allocation and zero steady-state per-token allocation.
The current host-reference bridge may return `choices`, but it must report
`lkjai_decode_backend=host_reference_recompute`,
`lkjai_kv_cache_backend=host_contiguous_bf16_diagnostic`, and
`lkjai_decode_supported=false`.

Sidecar metadata cannot promote host recompute. Accepted disclosure requires
generation to prefill real CUDA K/V tensors once and append one token per step
without calling host `transformer_forward` for the full prompt.

## Prompt And Tokenizer

The server loads artifact `tokenizer.json`, parses ordered OpenAI-style
`messages[]`, serializes them into the documented XML-like prompt format,
encodes prompt tokens, and decodes generated tokens back to assistant text.

The native tokenizer bridge implements the repo byte-level BPE subset needed by
the local tokenizer: added atomic tags, byte-level prompt text, BPE merges, and
special-token skipping on decode. Prompt serialization preserves raw message
content for training alignment, uses paired XML-like tags, and ends with
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
- Paged KV cache is a later batching optimization.
- Steady-state accepted decode must not allocate device memory per token.
- Host recompute choices are partial usability only and do not satisfy accepted
  decode even when diagnostic K/V allocation exists.

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

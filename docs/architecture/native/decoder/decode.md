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
- `usage.prompt_tokens`
- `usage.completion_tokens`
- `usage.total_tokens`

Dense and transformer artifacts continue to return HTTP `422` unsupported
decode with no `choices`.

## Prompt And Tokenizer

The server loads artifact `tokenizer.json`, serializes OpenAI-style chat
messages into the documented XML-like prompt format, encodes prompt tokens, and
decodes generated tokens back to assistant text.

The experimental P0 decode path may use a deterministic native byte-to-token
bridge until the full tokenizer bridge lands. Accepted decoder decode requires
the artifact tokenizer path above.

## Decode Loop

- Prefill consumes the prompt up to the configured context.
- Incremental decode appends one token at a time.
- The first implementation uses a native-owned contiguous KV cache.
- Paged KV cache is a later batching optimization.
- Steady-state decode must not allocate device memory per token.

## Sampling

Supported request controls:

- `max_tokens`
- `temperature`
- `top_k`
- `top_p`
- `seed`

Stop when the model emits `<eos>`, `</action>`, or the `max_tokens` cap.
Invalid or unsupported sampler values return JSON errors instead of silently
falling back.

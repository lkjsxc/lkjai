# Model Client Runtime

## Contract

- The web runtime must call a real OpenAI-compatible chat-completions endpoint.
- The runtime must not use policy-file fallbacks as production behavior.
- The model service owns model weights and CUDA lifecycle.
- The web service owns orchestration, tools, transcripts, and memory.

## Request/Response

- Request schema: `model`, `messages`, `max_tokens`, `temperature`.
- Response schema: first `choices[].message.content` is consumed.
- Each model step must return strict JSON action text for the agent parser.

## Failure Semantics

- HTTP request failures must surface as explicit transcript error events.
- Non-success status from model server must include status code and body text.
- Parse failures from model response must surface as explicit transcript errors.
- No canned fallback answer is allowed when model generation fails.

## Defaults

- `MODEL_API_URL=http://127.0.0.1:8081/v1/chat/completions`
- `MODEL_NAME=qwen3-1.7b-q4`
- `MODEL_MAX_NEW_TOKENS=512`
- `MODEL_TEMPERATURE=0.2`

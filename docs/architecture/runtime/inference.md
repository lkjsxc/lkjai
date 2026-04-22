# Model Client Runtime

## Loading

- Rust does not load production model weights in v1.
- Rust calls an OpenAI-compatible local model server.
- The Compose `model` service owns CUDA model loading.
- The `web` service owns agent orchestration, tools, memory, and transcripts.

## Generation

- The model client sends chat-completions requests.
- Agent prompts require one strict JSON action per model step.
- Canned model-status text is not a valid assistant response.
- Model errors become transcript `error` events.

## Defaults

- `MODEL_API_URL=http://127.0.0.1:8081/v1/chat/completions`.
- Compose overrides this to `http://model:8080/v1/chat/completions`.
- `MODEL_NAME=qwen3-1.7b-q4`.

# Model Client Runtime

## Loading

- Rust does not load production model weights in v1.
- Rust calls an OpenAI-compatible local model server.
- Rust can also load `policy://` trained policy artifacts for immediate local
  end-to-end operation.
- The Compose `model` service owns CUDA model loading.
- The `web` service owns agent orchestration, tools, memory, and transcripts.

## Generation

- The model client sends chat-completions requests.
- Agent prompts require one strict JSON action per model step.
- Canned model-status text is not a valid assistant response.
- Model errors become transcript `error` events.

## Defaults

- `MODEL_API_URL=policy:///app/data/train/policy/model.json`.
- Set `MODEL_API_URL=http://127.0.0.1:8081/v1/chat/completions` when using
  the optional llama.cpp model service.
- `MODEL_NAME=qwen3-1.7b-q4`.

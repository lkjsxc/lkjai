# Model Defaults

## Serving Model

- Default serving family: Qwen3 dense decoder.
- Default serving scale: 1.7B parameters.
- Default quantization: GGUF `Q4_K_M`.
- Default model artifact: `data/models/qwen3-1.7b-q4.gguf`.
- Compose model service loads that artifact as `/models/qwen3-1.7b-q4.gguf`.
- Default model server: llama.cpp OpenAI-compatible server.

## Tuning Model

- Default local tuning family: Qwen3 dense decoder.
- Default tuning scale: 0.6B parameters.
- Default tuning method: QLoRA.
- Larger 1.7B tuning is optional after 0.6B tuning passes fixtures.

## Runtime Context

- Model-native long context is not the memory mechanism.
- Active prompt context defaults to `4096` tokens.
- Operators may raise active context to `8192` tokens when VRAM allows.
- Old conversation state is represented by summaries and retrieved memory.

## Environment

- Compose `web` default: `http://model:8080/v1/chat/completions`.
- Host operator endpoint: `http://127.0.0.1:8081/v1/chat/completions`.
- Runtime requires a real model endpoint; policy-file fallback mode is removed.
- `MODEL_NAME` defaults to `qwen3-1.7b-q4`.
- `MODEL_CONTEXT_TOKENS=4096`.
- `MODEL_MAX_NEW_TOKENS=512`.
- `MODEL_TEMPERATURE=0.2`.

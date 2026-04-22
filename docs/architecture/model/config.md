# Model Defaults

## Serving Model

- Default serving family: Qwen3 dense decoder.
- Default serving scale: 1.7B parameters.
- Default quantization: GGUF `Q4_K_M`.
- Default model directory: `data/models/qwen3-1.7b-q4`.
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

- `MODEL_API_URL` may point to a trained policy with `policy://`.
- Compose defaults to `policy:///app/data/train/policy/model.json` for an
  immediately runnable trained local website.
- `MODEL_NAME` defaults to `qwen3-1.7b-q4`.
- `MODEL_CONTEXT_TOKENS=4096`.
- `MODEL_MAX_NEW_TOKENS=512`.
- `MODEL_TEMPERATURE=0.2`.

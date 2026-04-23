# Scratch Model Defaults

## Serving Model

- Default serving name: `lkjai-scratch-40m`.
- Default serving family: local scratch dense decoder.
- Default serving scale: 25-60M parameters.
- Default model artifact root: `data/models/lkjai-scratch-40m/`.
- Compose `inference` service loads scratch manifests from `/models`.
- Default serving direction: Python/Torch OpenAI-compatible inference runtime.

## Training Model

- Default training starts from random initialization.
- Default tokenizer is a locally trained byte-level BPE tokenizer.
- Default preset: `scratch-40m`.
- Default pipeline writes tokenizer, checkpoint, summary, and eval manifests.
- Pretrained bases and adapters are not default artifacts.

## Runtime Context

- Model-native long context is not the memory mechanism.
- Active prompt context defaults to `4096` tokens.
- Operators may raise active context to `8192` tokens when VRAM allows.
- Old conversation state is represented by summaries and retrieved memory.

## Environment

- Compose `web` default: `http://inference:8081/v1/chat/completions`.
- Host operator endpoint: `http://127.0.0.1:8081/v1/chat/completions`.
- Runtime requires a real model endpoint; policy-file fallback mode is removed.
- `MODEL_NAME` defaults to `lkjai-scratch-40m`.
- `MODEL_CONTEXT_TOKENS=4096`.
- `MODEL_MAX_NEW_TOKENS=512`.
- `MODEL_TEMPERATURE=0.2`.

# Local Deployment

## Scope

- v1 deployment is a local workstation deployment.
- Default bind is `127.0.0.1`.
- Host-YOLO makes public deployment unsafe.

## Start Inference + Web

```bash
cp .env.example .env
mkdir -p data/models/lkjai-scratch-20m data/train data/agent data/workspace
docker compose --profile inference up --build inference
docker compose --profile web up --build web
```

## Bootstrap Scratch Artifact

- The default artifact root is `data/models/lkjai-scratch-20m/`.
- Training export copies tokenizer, config, checkpoint, and serving manifests
  into that directory.
- Compose web uses `MODEL_API_URL=http://inference:8081/v1/chat/completions`.
- Host checks inference on `http://127.0.0.1:8081/v1/models`.
- Chat reports explicit model errors instead of dummy web-runtime responses.
- Default inference is Python/Torch until native Rust tensor decoding is ready.

## Rejected Bootstrap

- Do not download Qwen, Gemma, Kimi, DeepSeek, or any other pretrained model as
  the default runtime artifact.
- Do not bootstrap default serving from a GGUF pretrained model.
- Do not accept deterministic stub responses as model competency.

## Risk

- Do not expose the web port to an untrusted network.
- Do not run Host-YOLO with secrets mounted unless the operator accepts the
  risk.

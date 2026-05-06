# Local Deployment

## Scope

- current deployment is a local workstation deployment.
- Default bind is `127.0.0.1`.
- Host-YOLO makes public deployment unsafe.

## Start Runtime

```bash
cp .env.example .env
mkdir -p data/models/lkjai-scratch-40m data/train data/agent data/workspace
docker compose --profile web up --build web
```

## Bootstrap Scratch Artifact

- The default artifact root is `data/models/lkjai-scratch-40m/`.
- Training export copies tokenizer, config, checkpoint, and serving manifests
  into that directory.
- Compose web serves `/api/*` and `/v1/*` from one native process on
  `http://127.0.0.1:8080`.
- Host checks model readiness on `http://127.0.0.1:8080/v1/models`.
- Runtime configuration and `kjxlkj` adapter status are visible at
  `http://127.0.0.1:8080/api/config`.
- Chat reports explicit model errors instead of dummy web-runtime responses.
- Default inference is the native C++/CUDA server.

## Rejected Bootstrap

- Do not download Qwen, Gemma, Kimi, DeepSeek, or any other pretrained model as
  the default runtime artifact.
- Do not bootstrap default serving from a GGUF pretrained model.
- Do not accept deterministic stub responses as model competency.

## Risk

- Do not expose the web port to an untrusted network.
- Do not run Host-YOLO with secrets mounted unless the operator accepts the
  risk.

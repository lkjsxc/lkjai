# Local Deployment

## Scope

- current deployment is a local workstation deployment.
- Default bind is `127.0.0.1`.
- Host-YOLO makes public deployment unsafe.

## Start Runtime

```bash
cp .env.example .env
mkdir -p data/models/decoder-40m-3070 data/train data/agent data/workspace
docker compose --profile inference up --build -d
docker compose --profile sandbox up --build -d
docker compose --profile web up --build web
```

## Bootstrap Scratch Artifact

- The default artifact root is `data/models/decoder-40m-3070/`.
- Training export copies tokenizer, config, checkpoint, and serving manifests
  into that directory.
- Compose web serves static files on `http://127.0.0.1:8080`.
- Sandbox serves `/api/*` on `http://127.0.0.1:8082`.
- Inference serves `/v1/*` on `http://127.0.0.1:8081`.
- Host checks model readiness on `http://127.0.0.1:8081/v1/models`.
- Runtime configuration and `kjxlkj` adapter status are visible at
  `http://127.0.0.1:8082/api/config`.
- Chat reports explicit model errors instead of dummy web-runtime responses.
- Default inference is the native server. Dense artifacts provide readiness and
  diagnostics only; accepted decoder chat requires CUDA KV-cache evidence.

## Rejected Bootstrap

- Do not download Qwen, Gemma, Kimi, DeepSeek, or any other pretrained model as
  the default runtime artifact.
- Do not bootstrap default serving from a GGUF pretrained model.
- Do not accept deterministic stub responses as model competency.

## Risk

- Do not expose the web port to an untrusted network.
- Do not run Host-YOLO with secrets mounted unless the operator accepts the
  risk.

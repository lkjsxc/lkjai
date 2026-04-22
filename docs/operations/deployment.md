# Local Deployment

## Scope

- v1 deployment is a local workstation deployment.
- Default bind is `127.0.0.1`.
- Host-YOLO makes public deployment unsafe.

## Start Model + Web

```bash
cp .env.example .env
docker compose --profile model up -d model
docker compose --profile web up --build web
```

## Model Artifact

- Place GGUF model files under `data/models/`.
- Set `MODEL_GGUF` to the model path used by the model service.
- The Rust app talks to `MODEL_API_URL`.
- Chat reports explicit model errors instead of dummy assistant responses.

## Risk

- Do not expose the web port to an untrusted network.
- Do not run Host-YOLO with secrets mounted unless the operator accepts the
  risk.

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

## Bootstrap Model Artifact

```bash
mkdir -p data/models
curl -fL \
  "https://huggingface.co/lmstudio-community/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf" \
  -o data/models/qwen3-1.7b-q4.gguf
```

- Keep `MODEL_GGUF=qwen3-1.7b-q4.gguf` in `.env`.
- Compose model service reads `/models/${MODEL_GGUF}`.
- With defaults this resolves to `/models/qwen3-1.7b-q4.gguf`.
- Compose web uses `MODEL_API_URL=http://model:8080/v1/chat/completions`.
- Host checks the model endpoint on `http://127.0.0.1:8081/v1/models`.
- Chat reports explicit model errors instead of dummy assistant responses.

## Risk

- Do not expose the web port to an untrusted network.
- Do not run Host-YOLO with secrets mounted unless the operator accepts the
  risk.

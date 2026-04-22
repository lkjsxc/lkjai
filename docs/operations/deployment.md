# Local Deployment

## Scope

- v1 deployment is a local workstation deployment.
- Default bind is `127.0.0.1`.
- Host-YOLO makes public deployment unsafe.

## Start

```bash
cp .env.example .env
docker compose --profile web up --build web
```

## Model Artifact

- Place serving exports under `data/train/models/lkj-150m`, or set
  `MODEL_DIR` to another complete export directory.
- A serving export must include `config.json`, `model.safetensors`, and
  `tokenizer.json`.
- The export must also include `manifest.json`; runtime validates the
  co-located tokenizer hash before loading weights.
- `INFERENCE_DEVICE=cuda` is the default. Use `cpu` or `auto` only when CUDA
  fallback is intentional.
- The app remains bootable without an export, but chat reports an explicit
  model load error instead of a dummy assistant response.

## Risk

- Do not expose the web port to an untrusted network.
- Do not run Host-YOLO with secrets mounted unless the operator accepts the
  risk.

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

- Place serving exports under `data/models/lkj-150m`.
- The app remains bootable without an export and reports missing-model status in
  chat transcripts.

## Risk

- Do not expose the web port to an untrusted network.
- Do not run Host-YOLO with secrets mounted unless the operator accepts the
  risk.

# Compose Contract

## Profiles

- `model`: llama.cpp CUDA model server.
- `web`: Rust axum agent orchestrator.
- `train`: CUDA post-training container.
- `verify`: repository verification container.

## Data Mount

- All profiles mount `./data:/app/data`.
- Model serving reads GGUF files from `/app/data/models`.
- Training writes datasets, adapters, exports, and logs under `/app/data/train`.
- Web writes transcripts and memory under `/app/data/agent`.

## GPU

- `model` requests NVIDIA GPU access.
- `train` requests NVIDIA GPU access for QLoRA runs.
- `web` does not load model weights and does not require CUDA.

## Commands

```bash
docker compose --profile web up --build
docker compose --profile train up --build train
docker compose --profile verify build verify
docker compose --profile verify run --rm verify
```

## Training Defaults

- The `train` service defaults to `TRAIN_PRESET=quick`.
- `TRAIN_PRESET=agent` enables RTX 3070 QLoRA defaults.
- `TRAIN_FIXED_EVAL_THRESHOLD` defaults to `0.80`.
- `TRAIN_ENFORCE_COMPETENCY` defaults to disabled for quick runs.
- Training writes to `TRAIN_DATA_DIR`, default `/app/data/train`.

## Presets

- `quick`: deterministic fixture-scale validation and eval.
- `agent`: QLoRA-oriented tuning defaults for RTX 3070 8GB.
- `custom`: all behavior controlled by explicit `TRAIN_*` environment values.

## Long-Run Contract Links

- [training/long-run.md](training/long-run.md)
- [training/competency-gate.md](training/competency-gate.md)

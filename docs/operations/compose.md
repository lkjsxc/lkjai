# Compose Contract

## Profiles

- `model`: llama.cpp CUDA model server.
- `web`: Rust axum agent orchestrator.
- `train`: CUDA training container (Transformers + PEFT + bitsandbytes).
- `verify`: repository verification container.

## Data Mount

- All profiles mount `./data:/app/data`.
- Model serving mounts `./data/models` to `/models` when profile `model` is
  enabled.
- Model serving loads `/models/${MODEL_GGUF}`.
- Training writes datasets, adapter checkpoints, exports, and logs under
  `/app/data/train`.
- Web writes transcripts and memory under `/app/data/agent`.

## GPU

- `model` requests NVIDIA GPU access.
- `train` requests NVIDIA GPU access for QLoRA runs.
- `web` does not load model weights and does not require CUDA.

## Healthcheck

- The `model` service includes a Docker healthcheck using `curl -f
  http://localhost:8080/health` or `/v1/models`.
- The `web` service may optionally wait for `model` to be healthy.

## Commands

```bash
mkdir -p data/models
curl -fL \
  "https://huggingface.co/lmstudio-community/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf" \
  -o data/models/qwen3-1.7b-q4.gguf
```

```bash
docker compose --profile model up -d model
docker compose --profile web up --build web
docker compose --profile train up --build train
docker compose --profile verify build verify
docker compose --profile verify run --rm verify
```

## Training Defaults

- The `train` service defaults to `TRAIN_PRESET=agent`.
- `TRAIN_PRESET=quick` runs real training with reduced steps for fast checks.
- `TRAIN_FIXED_EVAL_THRESHOLD` defaults to `0.80`.
- `TRAIN_ENFORCE_COMPETENCY` defaults to disabled unless explicitly enabled.
- Training writes to `TRAIN_DATA_DIR`, default `/app/data/train`.

## Presets

- `quick`: real training with small step count for CI smoke.
- `agent`: QLoRA-oriented tuning defaults for RTX 3070 8GB.
- `custom`: all behavior controlled by explicit `TRAIN_*` environment values.

## Long-Run Contract Links

- [training/long-run.md](training/long-run.md)
- [training/competency-gate.md](training/competency-gate.md)

# Compose Contract

## Profiles

- `inference`: native OpenAI-compatible scratch inference service.
- `web`: Rust axum agent orchestrator plus its inference dependency.
- `train`: native scratch training container.
- `verify`: repository verification container.

## Data Mount

- All runtime profiles mount `./data:/app/data`.
- Inference mounts `./data/models` to `/models`.
- Inference loads `/models/${MODEL_NAME}`.
- The `web` profile also activates `inference`.
- Web waits for inference process health before serving traffic.
- Model readiness is reported separately through `/api/model` and
  `GET /v1/models`.
- Inference loads exported scratch checkpoints and generates actions directly.
- Inference must not use exact supervised lookup, prompt matching, or canned
  response tables.
- Training writes datasets, tokenizer, checkpoints, exports, and logs under
  `/app/data/train`.
- Web writes transcripts and memory under `/app/data/agent`.
- Web uses `/app/data/workspace` as the only filesystem root for tools.
- Web must not mount the host root.

## GPU

- `train` requests NVIDIA GPU access for scratch training.
- `inference` requests NVIDIA GPU access for scratch serving.
- `inference` falls back to CPU only as a visible degraded mode.
- `/api/model` and the web UI must show CUDA availability and active device.
- CPU fallback is acceptable for development but is not an acceptable quality or
  latency baseline.
- `web` does not load model weights and does not require CUDA.

## Commands

```bash
cp .env.example .env
mkdir -p data/models/lkjai-scratch-40m data/train data/agent data/workspace
docker compose --profile inference up --build inference
docker compose --profile web up --build web
docker compose --profile train up --build train
docker compose --progress quiet --profile verify up --build --abort-on-container-exit verify
```

## Compact Output

- Prefer `--progress quiet` for Compose builds when an LLM agent is reading the
  result.
- For long-running services, inspect bounded logs with
  `docker compose logs --tail=120 SERVICE`.
- `ops/verify.sh` stores full check logs under `/tmp/lkjai-verify-logs` and prints a
  compact pass/fail summary.

## Training Defaults

- The `train` service runs `lkjai-native-train`.
- Training writes to `TRAIN_DATA_DIR`, default `/app/data/train`.
- The default Compose command is a two-step smoke run.
- Long native training must save `lkjai-native-artifact-v1` under `data/models`.

## Presets

- `smoke`: tiny native run for local verification.
- `agent`: `scratch-40m` defaults for RTX 3070 8GB research.

## Long-Run Contract Links

- [training/long-run.md](training/long-run.md)
- [training/competency-gate.md](training/competency-gate.md)

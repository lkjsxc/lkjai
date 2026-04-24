# Compose Contract

## Profiles

- `inference`: Python/Torch OpenAI-compatible scratch inference service.
- `web`: Rust axum agent orchestrator.
- `train`: PyTorch scratch training container.
- `verify`: repository verification container.

## Data Mount

- All runtime profiles mount `./data:/app/data`.
- Inference mounts `./data/models` to `/models`.
- Inference loads `/models/${MODEL_NAME}`.
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
- `inference` loads exported PyTorch scratch checkpoints and tokenizers.
- `web` does not load model weights and does not require CUDA.

## Commands

```bash
cp .env.example .env
mkdir -p data/models/lkjai-scratch-60m data/train data/agent data/workspace
docker compose --profile inference up --build inference
docker compose --profile web up --build web
docker compose --profile train up --build train
docker compose --profile verify up --build --abort-on-container-exit verify
```

## Training Defaults

- The `train` service defaults to `TRAIN_PRESET=agent`.
- `TRAIN_PRESET=quick` runs tiny scratch training with reduced steps.
- `TRAIN_FIXED_EVAL_THRESHOLD` defaults to `0.60` for artifact reporting.
- `TRAIN_BEHAVIORAL_THRESHOLD` defaults to `0.35` for the next pass-rate ladder.
- `TRAIN_ENFORCE_COMPETENCY` defaults to disabled unless explicitly enabled.
- Training writes to `TRAIN_DATA_DIR`, default `/app/data/train`.
- Behavioral competency requires `data/train/runs/behavioral-eval.json`
  `pass_rate >= 0.80`.

## Presets

- `quick`: tiny scratch run for local smoke.
- `agent`: `scratch-60m` defaults for RTX 3070 8GB research.
- `custom`: all behavior controlled by explicit `TRAIN_*` environment values.

## Long-Run Contract Links

- [training/long-run.md](training/long-run.md)
- [training/competency-gate.md](training/competency-gate.md)

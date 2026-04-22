# Compose Contract

## Profiles

- `train`: CUDA PyTorch training container.
- `web`: Rust axum web container.
- `verify`: repository verification container.

## Data Mount

- All profiles mount `./data:/app/data`.
- Training writes corpora, tokenizers, checkpoints, exports, and logs under
  `/app/data/train`.
- Web reads the trained export from `/app/data/train/models/lkj-150m` by
  default and writes agent transcripts under `/app/data`.

## GPU

- `train` requests NVIDIA GPU access.
- `web` builds Candle with CUDA support and defaults to
  `INFERENCE_DEVICE=cuda`.
- `CUDA_COMPUTE_CAP` defaults to `86` for RTX 30-series CUDA builds; set it
  before building if the serving GPU has a different compute capability.
- CPU fallback is explicit only with `INFERENCE_DEVICE=cpu` or `auto`.

## Commands

```bash
docker compose --profile train up --build
docker compose --profile web up --build web
docker compose --profile verify build verify
docker compose --profile verify run --rm verify
```

## Training Defaults

- The `train` service defaults to `TRAIN_PRESET=longrun`.
- The default long-run target is `TRAIN_MAX_DURATION_SECS=21600` (~6 hours).
- `TRAIN_STEPS` is still required as a minimum step floor before duration stop.
- `TRAIN_FIXED_EVAL_THRESHOLD` defaults to `0.80`.
- `TRAIN_ENFORCE_COMPETENCY` defaults to enabled for long-run acceptance.
- `TRAIN_TOKENIZER_SAMPLE_CHARS` defaults to `5000000` to bound tokenizer RAM.
- Training writes to `TRAIN_DATA_DIR`, default `/app/data/train`.

## Presets

- `longrun`: full dataset/model defaults plus duration-aware stop and competency enforcement.
- `quick`: deterministic fixture-scale run for fast local debugging.
- `full`: full dataset/model defaults with configurable stop knobs.
- `custom`: all behavior controlled by explicit `TRAIN_*` environment values.

## Long-Run Contract Links

- [training/long-run.md](training/long-run.md)
- [training/competency-gate.md](training/competency-gate.md)

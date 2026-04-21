# Compose Contract

## Profiles

- `train`: CUDA PyTorch training container.
- `web`: Rust axum web container.
- `verify`: repository verification container.

## Data Mount

- All profiles mount `./data:/app/data`.
- Training writes corpora, tokenizers, checkpoints, exports, and logs under
  `/app/data`.
- Web reads model exports and writes agent transcripts under `/app/data`.

## GPU

- `train` requests NVIDIA GPU access.
- `web` requests NVIDIA GPU access for Candle inference when available.
- CPU fallback is allowed for smoke verification only.

## Commands

```bash
docker compose --profile train up --build
docker compose --profile web up --build web
docker compose --profile verify run --rm verify
```

## Training Defaults

- The `train` service starts the `quick` preset by default so
  `docker compose --profile train up --build` completes on ordinary hosts.
- Training writes to `TRAIN_DATA_DIR`, defaulting to `/app/data/train`, so quick
  runs do not overwrite the serving export.
- Use `TRAIN_PRESET=full` for the full `lkj-150m` config and large corpus.
- Use `TRAIN_PRESET=custom` with `TRAIN_TINY`, `TRAIN_TOKEN_BUDGET`,
  `TRAIN_STEPS`, and related knobs for explicit experiments.

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

- The `train` service starts the full corpus, tokenizer, model, and export
  pipeline by default.
- Training can be scaled down for checks with `TRAIN_TINY=1`,
  `TRAIN_TOKEN_BUDGET`, and `TRAIN_STEPS`.

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
docker compose --profile train run --rm train smoke
docker compose --profile web up --build web
docker compose --profile verify run --rm verify
```

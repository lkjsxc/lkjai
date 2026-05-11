# Compose Contract

Owner: `docs/operations/compose.md`.
State: canonical Compose profile, mount, port, and verification contract.

## Profiles

- `inference`: current native OpenAI-compatible scratch inference service.
- `web`: merged native server with `/api/*` runtime routes and `/v1/*`
  inference routes.
- `train`: native scratch training container.
- `corpus`: isolated public corpus acquisition and JSONL preparation container.
- `verify`: repository verification container.

## Data Mount

- All runtime profiles mount `./data:/app/data`.
- Inference mounts `./data/models` to `/models`.
- Inference loads `/models/${MODEL_NAME}`.
- The `web` profile does not start a second model service.
- The `inference` profile remains as a direct `/v1/*` diagnostic service.
- The merged server process owns both `/api/*` and `/v1/*` routes.
- Model readiness is reported separately through `/api/model` and
  `GET /v1/models`.
- Inference loads exported native artifacts. Dense and transformer artifacts
  return HTTP `422` unsupported with no `choices`; decoder artifacts can return
  choices when exported with the real local tokenizer.
- Inference must not use exact supervised lookup, prompt matching, or canned
  response tables.
- Training writes datasets, tokenizer, checkpoints, exports, and logs under
  `/app/data/train`.
- Training mounts committed configs at `/workspace/configs`.
- Corpus acquisition writes raw public snapshots under
  `/app/data/raw/cosmopedia` and prepared JSONL under
  `/app/data/public-corpus`.
- Corpus acquisition mounts the Hugging Face token reference from
  `HF_SECRET_FILE`, default `./data/secrets/hf_token`, read-only at
  `/run/secrets/hf_token_source`. The file may contain a raw token or a local
  operator note with the token value, but docs, manifests, and committed config
  must contain only paths or environment variable names.
- `HF_TOKEN` and `HF_TOKEN_FILE` may override the mounted token source for local
  runs. Do not commit token values.
- Hugging Face CLI, Python, and Arrow dependencies are allowed only in the
  `corpus` image, not in `train`, `web`, `inference`, or `verify`.
- Web writes transcripts and memory under `/app/data/agent`.
- Web uses `/app/data/workspace` as the only filesystem root for tools.
- Web must not mount the host root.
- The merged server uses `KJXLKJ_USER` and `KJXLKJ_BEARER_TOKEN` for typed
  `/api/users/{user}/resources/...` calls.
- `GET /api/config` reports those settings and keeps mutable resource tools
  disabled until confirmation-gated tool execution is implemented.

## GPU

- `train` requests NVIDIA GPU access for scratch training.
- `inference` requests NVIDIA GPU access for scratch serving.
- `inference` falls back to CPU only as a visible degraded mode.
- `/api/model` and the web UI must show CUDA availability and active device.
- CPU fallback is acceptable for development but is not an acceptable quality or
  latency baseline.
- `web` loads model weights through the merged native server and requests CUDA.

## Commands

```bash
cp .env.example .env
mkdir -p data/models/lkjai-scratch-40m data/train data/agent data/workspace
docker compose --profile inference up --build inference
docker compose --profile web up --build web
docker compose --profile train up --build train
docker compose --profile corpus run --build --rm corpus download-public-pretrain
docker compose --progress quiet --profile verify run --build --rm verify
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
- `TRAIN_CONFIG` selects the training-run JSON config.
- `TRAIN_NATIVE_CONFIG` selects the native model-shape JSON config.
- Long native training must save `lkjai-native-artifact` under `data/models`.
- The `verify` service requires NVIDIA GPU access and builds native code with
  the real CUDA compiler.

## Corpus Defaults

- The `corpus` service owns Hugging Face acquisition and public-pretrain JSONL
  materialization.
- `download-public-pretrain` reads `corpus/sources/public-pretrain.json`, calls
  the Hugging Face parquet API for active Cosmopedia train files, and downloads
  them under `TRAIN_PUBLIC_DATA_DIR`.
- `prepare-public-pretrain` emits text-only `train`, `val`, and `holdout` JSONL
  shards under `TRAIN_CORPUS_DIR` until `TRAIN_PUBLIC_PRETRAIN_TOKENS`.
- `validate-public-pretrain` checks manifests and generated rows for text-only
  provenance, pinned source revisions, checksums, and split counts.
- `build-public-pretrain-cache` runs the native `lkjai-native-packed-cache`
  binary on `TRAIN_PACKED_CACHE_SOURCE`, defaulting to the first train shard.
  It requires `TRAIN_TOKENIZER_JSON` to point at the local byte-level BPE
  tokenizer; smoke/export tokenizers are not valid public cache inputs.

## Presets

- `smoke`: tiny native run for local verification.
- `agent`: `scratch-40m` defaults for RTX 3070 8GB research.

## Long-Run Contract Links

- [training/runbooks/long-run.md](training/runbooks/long-run.md)
- [training/gates/competency-gate.md](training/gates/competency-gate.md)

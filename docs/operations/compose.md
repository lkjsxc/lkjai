# Compose Contract

Owner: `docs/operations/compose.md`.
State: canonical Compose profile, mount, port, and verification contract.

## Profiles

- `inference`: current native OpenAI-compatible scratch inference service on
  `http://127.0.0.1:8081`.
- `sandbox`: native agent API runtime on `http://127.0.0.1:8082`.
- `web`: static frontend on `http://127.0.0.1:8080`.
- `train`: native scratch training container.
- `corpus`: isolated public corpus acquisition and JSONL preparation container.
- `verify`: repository verification container.

## Data Mount

- `sandbox`, `inference`, and `train` mount `./data:/app/data`.
- `verify` uses `/tmp/lkjai-verify-data` inside the container and must not
  write through to host `./data`.
- `web` mounts no model, data, workspace, or GPU paths.
- Inference mounts `./data/models` to `/models`.
- Inference loads `/models/${MODEL_NAME}`.
- Training exports under `/app/data/models/${TRAIN_MODEL_NAME}` by default;
  serving still selects artifacts with `MODEL_NAME`.
- The `web` profile serves static files only.
- The `sandbox` profile calls `MODEL_API_URL`, default
  `http://inference:8081/v1/chat/completions`.
- The `inference` profile remains as the direct `/v1/*` OpenAI-compatible API
  service. It rejects `/api/*` and frontend routes.
- The sandbox process owns `/api/*` routes and rejects `/v1/*` and frontend
  routes.
- Model readiness is reported through sandbox `/api/model` and direct
  inference `GET /v1/models`.
- Inference loads exported native artifacts. Dense and transformer artifacts
  return HTTP `422` unsupported with no `choices`; decoder artifacts can return
  OpenAI-compatible `choices` when exported with the real local tokenizer.
  Accepted CUDA decode disclosure requires the decoder evidence gate in
  [decoder/acceptance.md](../architecture/native/decoder/acceptance.md).
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
- Sandbox writes transcripts and memory under `/app/data/agent`.
- Sandbox uses `/app/data/workspace` as the only filesystem root for tools.
- Sandbox mounts read-only source context into that workspace for `fs.read`:
  `docs`, `native`, `web`, `ops`, `scripts`, `configs`, `training`,
  `corpus`, root `README.md`, `Dockerfile.web`, and `compose.yaml`.
- Sandbox must not mount the host root.
- The sandbox uses `KJXLKJ_USER` and `KJXLKJ_BEARER_TOKEN` for typed
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
- `web` never loads model weights and requests no GPU.

## Commands

```bash
cp .env.example .env
mkdir -p \
  data/models/dense-diagnostic-scratch-40m \
  data/models/decoder-40m-3070 \
  data/train data/agent data/workspace
docker compose --profile inference up --build -d
docker compose --profile sandbox up --build -d
docker compose --profile web up --build web
docker compose --profile corpus run --build --rm corpus download-public-pretrain
docker compose --profile corpus run --build --rm corpus build-tokenizer
docker compose --profile corpus run --rm corpus validate-public-pretrain
docker compose --profile corpus run --rm corpus build-public-pretrain-cache
docker compose --profile train up --build train
docker compose --progress quiet --profile verify run --build --rm verify
```

Use this detached command for direct OpenAI-compatible chat:

```bash
docker compose --profile inference up --build -d
```

For chat, `.env` must point `MODEL_NAME` at an existing decoder export:

```dotenv
MODEL_NAME=decoder-40m-3070
```

Minimal API probes:

```bash
curl --fail http://127.0.0.1:8081/v1/models
curl -sS -X POST http://127.0.0.1:8081/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "decoder-40m-3070",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

Expected chat result depends on the artifact kind: decoder artifacts return
`choices`; dense and transformer artifacts return HTTP `422` without `choices`.
If `data/models/${MODEL_NAME}` is missing, `GET /v1/models` returns HTTP `503`.

For a browser chat-attempt run that overwrites `decoder-40m-3070` without
claiming acceptance, first build the tokenizer and short-window SFT packed
cache:

```bash
docker compose --profile corpus run --build --rm corpus build-tokenizer

docker compose --profile corpus run --build --rm corpus lkjai-native-packed-cache build \
  --source /app/data/corpus/generated/kimi-sft-60m/train \
  --tokenizer /app/data/train/tokenizer/tokenizer.json \
  --config /workspace/configs/native/decoder_40m_bf16_3070.json \
  --out /app/data/train/datasets/packed/train-assistant_masked_sft-seq128 \
  --split train \
  --objective assistant_masked_sft \
  --seq-len 128 \
  --run-id decoder-40m-chat-attempt-4h
```

Then run an isolated one-step start check before spending the full four hours:

```bash
TRAIN_CONFIG=/workspace/configs/training/decoder_4h_chat_attempt_3070.json \
TRAIN_MODEL_NAME=decoder-40m-3070 \
TRAIN_RUN_PURPOSE=chat_attempt \
TRAIN_DATA_DIR=/app/data/chat-attempt-start-check/train \
TRAIN_MAX_OPTIMIZER_STEPS=1 \
TRAIN_TARGET_SECONDS=0 \
docker compose --profile train run --build --rm train --train --mode decoder
```

Then run the four-hour non-acceptance training lane:

```bash
TRAIN_CONFIG=/workspace/configs/training/decoder_4h_chat_attempt_3070.json \
TRAIN_MODEL_NAME=decoder-40m-3070 \
TRAIN_RUN_PURPOSE=chat_attempt \
docker compose --profile train up --build train
```

Start the model, sandbox, and web UI:

```bash
MODEL_NAME=decoder-40m-3070 docker compose --profile sandbox up --build -d
docker compose --profile web up --build -d web
```

Before opening the browser, probe:

```bash
curl --fail http://127.0.0.1:8081/v1/models
curl --fail http://127.0.0.1:8082/healthz
curl --fail http://127.0.0.1:8082/api/model
```

Open `http://127.0.0.1:8080`. A valid chat attempt shows either assistant
content or a visible failure stop reason from `/api/chat`; this lane remains
non-accepted and should continue to disclose `lkjai_decode_accepted=false`.

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
- The default Compose command is `--train --mode decoder`, a real decoder 40M
  training run bounded by the committed two-hour config.
- `TRAIN_MODEL_NAME` selects the training export name and defaults to
  `decoder-40m-3070`; it is independent from serving `MODEL_NAME`.
- `TRAIN_CONFIG` selects the training-run JSON config.
- `TRAIN_NATIVE_CONFIG` selects the native model-shape JSON config.
- `TRAIN_TARGET_SECONDS` can override the committed wall-clock deadline.
- The train profile consumes an existing packed cache. It does not build or
  repair tokenizer/cache data; missing, incompatible, or smoke-fixture caches
  fail before training.
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
- `build-tokenizer` writes `TRAIN_TOKENIZER_JSON`, default
  `/app/data/train/tokenizer/tokenizer.json`, with a deterministic native
  byte-level BPE-compatible tokenizer and atomic canonical XML-like tags.
- `build-public-pretrain-cache` runs the native `lkjai-native-packed-cache`
  binary on `TRAIN_PACKED_CACHE_SOURCE`, defaulting to the prepared train
  shard directory. It streams one JSONL file or sorted `*.jsonl` shards into
  `TRAIN_PACKED_CACHE_DIR`, default
  `/app/data/train/datasets/packed/train-causal_lm_full-seq1024`.
- Public-pretrain cache validation must compare the cache against the exact
  source path, `TRAIN_TOKENIZER_JSON`, and
  `configs/native/decoder_40m_bf16_3070.json`; smoke/export tokenizers are not
  valid public cache inputs.

## Presets

- `smoke`: tiny native run for local verification.
- `agent`: `scratch-40m` defaults for RTX 3070 8GB research.

## Long-Run Contract Links

- [training/runbooks/long-run.md](training/runbooks/long-run.md)
- [training/gates/competency-gate.md](training/gates/competency-gate.md)

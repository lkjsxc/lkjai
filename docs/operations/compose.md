# Compose Contract

## Profiles

- `inference`: Python/Torch OpenAI-compatible scratch inference service.
- `web`: Rust axum agent orchestrator plus its inference dependency.
- `train`: PyTorch scratch training container.
- `corpus`: CPU public-corpus materialization container.
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
- Training reads the active public pretraining corpus from
  `/app/data/public-corpus`
  and writes datasets, tokenizer, checkpoints, exports, and logs under
  `/app/data/train`.
- Corpus materialization reads user-downloaded raw files from
  `/app/data/raw/cosmopedia`.
- Corpus download can read `HF_TOKEN` or `HF_TOKEN_FILE`; token values must not
  be written to reports or logs.
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
docker compose --profile corpus run --rm corpus download-public-pretrain
docker compose --profile corpus run --rm corpus prepare-public-pretrain
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

- The `train` service defaults to `TRAIN_PRESET=agent` with
  `TRAIN_CONFIG=/workspace/configs/training/scratch_40m_12h.json`.
- `TRAIN_PRESET=quick` runs tiny scratch training with reduced steps.
- `TRAIN_OBJECTIVE` defaults to `causal_lm_full`; use
  `assistant_masked_sft` for XML-action SFT.
- `TRAIN_CORPUS_DIR` defaults to `/app/data/public-corpus`.
- `TRAIN_PUBLIC_DATA_DIR` defaults to `/app/data/raw/cosmopedia`.
- `TRAIN_PUBLIC_PRETRAIN_TOKENS` defaults to `440000000`.
- `TRAIN_FIRST_PARTY_SFT_TOKENS` defaults to `60000000`.
- `TRAIN_FIXED_EVAL_THRESHOLD` defaults to `0.60` for artifact reporting.
- `TRAIN_BEHAVIORAL_THRESHOLD` defaults to `0.35` for the next pass-rate ladder.
- `TRAIN_ENFORCE_COMPETENCY` defaults to disabled unless explicitly enabled.
- Training writes to `TRAIN_DATA_DIR`, default `/app/data/train`.
- Behavioral competency requires `data/train/runs/behavioral-eval.json`
  `pass_rate >= 0.80`.
- Default `TRAIN_CORPUS_SIZE` is `120000` and `TRAIN_MAX_STEPS` is `400000`
  optimizer steps.

## Presets

- `quick`: tiny scratch run for local smoke.
- `agent`: `scratch-40m` defaults for RTX 3070 8GB research.
- `custom`: all behavior controlled by explicit `TRAIN_*` environment values.

## Long-Run Contract Links

- [training/long-run.md](training/long-run.md)
- [training/competency-gate.md](training/competency-gate.md)

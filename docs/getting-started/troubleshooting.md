# Troubleshooting

## Goal

Recover from common setup and runtime failures.

## Inference Not Reachable

Symptom: `GET /api/model` shows `reachable: false`.

Check:
```bash
docker compose --profile inference ps
docker logs <inference-container-name>
curl -v http://127.0.0.1:8081/v1/models
```

Fix:
- Ensure `MODEL_NAME=lkjai-scratch-40m` unless testing another artifact.
- Ensure `data/models/${MODEL_NAME}` exists or run training/export first.
- Run `docker compose --profile web up --build web` for the browser app.
- Run `docker compose --profile inference up --build inference` only for
  model-server-only checks.

## Web Runtime Cannot Reach Inference

Symptom: Chat returns `model_error` events.

Check:
- `MODEL_API_URL` in the web container matches the inference container address.
- Inside the web container: `curl http://inference:8081/v1/models`.

Fix:
- If using Docker Compose, start with `--profile web`; it starts inference too.
- Use
  `http://inference:8081/v1/chat/completions`.
- If using host networking outside Compose, use the host IP instead.

## Training Finishes Instantly

Symptom: checkpoint artifacts exist but metrics are missing or empty.

Check:
- `TRAIN_PRESET` is not accidentally set to a tiny custom run.
- Dataset row count: `wc -l data/train/datasets/corpus.jsonl`.
- Training logs for loss output.

Fix:
- Use `TRAIN_PRESET=agent`.
- Run the native train profile first.
- Verify `data/train/datasets/corpus.jsonl` has >= 100 rows.

## Verify Profile Fails

Symptom: `docker compose --profile verify up --build --abort-on-container-exit verify`
exits non-zero.

Check:
- Rust formatting: `cargo fmt -- --check`.
- Rust tests: `cargo test`.
- Native tests: `ctest --test-dir /tmp/lkjai-native-build --output-on-failure`.
- Line limits: `cargo run --bin lkjai -- quality check-lines`.
- Docs topology: `cargo run --bin lkjai -- docs validate-topology`.

Fix each failing gate before retrying.

## Out Of Memory During Training

Symptom: CUDA OOM or container killed.

Fix:
- Reduce `TRAIN_SEQUENCE_LEN`.
- Reduce `TRAIN_HIDDEN_SIZE`.
- Reduce `TRAIN_LAYERS`.
- Increase `TRAIN_GRADIENT_ACCUMULATION` and reduce batch size.
- Use `TRAIN_PRESET=quick` to verify the pipeline before a long run.

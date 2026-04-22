# Troubleshooting

## Goal

Recover from common setup and runtime failures.

## Model Server Not Reachable

Symptom: `GET /api/model` shows `reachable: false`.

Check:
```bash
docker compose --profile model ps
docker logs <model-container-name>
curl -v http://127.0.0.1:8081/v1/models
```

Fix:
- Ensure `data/models/${MODEL_GGUF}` exists.
- Ensure NVIDIA container runtime is installed.
- Run `docker compose --profile model up -d model`.

## Web Runtime Cannot Reach Model

Symptom: Chat returns `model_error` events.

Check:
- `MODEL_API_URL` in the web container matches the model container address.
- Inside the web container: `curl http://model:8080/v1/models`.

Fix:
- If using host networking, use host IP.
- If using Docker Compose default network, use `http://model:8080/v1/chat/completions`.

## Training Finishes Instantly

Symptom: No GPU activity, adapter directory only contains JSON markers.

Check:
- `TRAIN_PRESET` is not `quick` for real runs.
- Dataset row count: `wc -l data/train/datasets/fixtures.jsonl`.
- Training logs for `trainer.train()` output.

Fix:
- Use `TRAIN_PRESET=agent`.
- Run `python -m lkjai_train.cli prepare-corpus` first.
- Verify `data/train/datasets/corpus.jsonl` has >= 100 rows.

## Verify Profile Fails

Symptom: `docker compose --profile verify run --rm verify` exits non-zero.

Check:
- Rust formatting: `cargo fmt -- --check`.
- Rust tests: `cargo test`.
- Python tests: `python3 -m pytest training/tests`.
- Line limits: `cargo run --bin lkjai -- quality check-lines`.
- Docs topology: `cargo run --bin lkjai -- docs validate-topology`.

Fix each failing gate before retrying.

## Out of Memory During Training

Symptom: CUDA OOM or container killed.

Fix:
- Reduce `TRAIN_SEQUENCE_LEN`.
- Reduce `TRAIN_LORA_RANK`.
- Increase `TRAIN_GRADIENT_ACCUMULATION` and reduce batch size.
- Ensure `TRAIN_LOAD_IN_4BIT=1`.

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
- Ensure `MODEL_NAME` points to an existing artifact under
  `data/models/${MODEL_NAME}`.
- Ensure `data/models/${MODEL_NAME}` exists or run training/export first.
- Run `docker compose --profile web up --build web` for the merged runtime and
  model server.
- Run `docker compose --profile inference up --build -d` for direct
  OpenAI-compatible `/v1/*` checks.

For direct chat, use a decoder export, for example:

```dotenv
MODEL_NAME=decoder-2h-40m-3070
```

## Chat Returns Model Errors

Symptom: Chat returns `model_error` events.

Check:
- `curl http://127.0.0.1:8080/api/model`.
- `curl http://127.0.0.1:8080/v1/models`.

Fix:
- Ensure `data/models/${MODEL_NAME}` exists.
- Use a decoder artifact for OpenAI-compatible `choices` on
  `http://127.0.0.1:8081/v1/chat/completions`. Dense and transformer artifacts
  return HTTP `422` without `choices`.
- Accepted chat disclosure requires CUDA KV-cache decode evidence.

## Training Finishes Instantly

Symptom: checkpoint artifacts exist but metrics are missing or empty.

Check:
- Dataset row count: `wc -l data/train/datasets/corpus.jsonl`.
- Training logs for loss output.

Fix:
- Run the native train profile first.
- Verify `data/train/datasets/corpus.jsonl` has >= 100 rows.

## Verify Profile Fails

Symptom: `docker compose --progress quiet --profile verify run --build --rm verify`
exits non-zero.

Check:
- Native tests: `ctest --test-dir /tmp/lkjai-native-build --output-on-failure`.
- Line limits: `lkjai-native-repo-check line-limits --repo /workspace`.
- Docs topology: `lkjai-native-repo-check docs-topology --repo /workspace`.
- Links: `lkjai-native-repo-check docs-links --repo /workspace`.

Fix each failing gate before retrying.

## Out Of Memory During Training

Symptom: CUDA OOM or container killed.

Fix:
- Reduce `TRAIN_SEQUENCE_LEN`.
- Reduce `TRAIN_HIDDEN_SIZE`.
- Reduce `TRAIN_LAYERS`.
- Increase `TRAIN_GRADIENT_ACCUMULATION` and reduce batch size.
- Use `docker compose --profile train run --rm train --smoke --steps 2` for a
  smoke pipeline check before a long run.

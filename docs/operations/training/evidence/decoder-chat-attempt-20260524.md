# Decoder Chat Attempt

Owner: `docs/operations/training/evidence/decoder-chat-attempt-20260524.md`.
State: evidence note.

## Summary

The non-acceptance `decoder-40m-3070` chat-attempt lane completed on the local
RTX 3070 on 2026-05-24. It used `assistant_masked_sft` data and intentionally
served through the same model name so the browser path could exercise a real
decoder artifact without claiming acceptance.

## Training Inputs

- training config: `configs/training/decoder_4h_chat_attempt_3070.json`
- run purpose: `chat_attempt`
- target seconds: `14400`
- packed cache:
  `data/train/datasets/packed/train-assistant_masked_sft-seq1024`
- served artifact: `data/models/decoder-40m-3070`

## Result

`data/train/runs/train-report.json` reported:

```json
{
  "status": "success",
  "run_purpose": "chat_attempt",
  "accepted_cuda_training": false,
  "target_seconds": 14400,
  "deadline_hit": true,
  "stop_reason": "wall_clock_deadline",
  "optimizer_steps": 6,
  "loss_finite": true,
  "decode_supported": true,
  "logits_check_passed": true
}
```

The served manifest exists at `data/models/decoder-40m-3070/manifest.json` and
reports `artifact_kind=export`, `kind=decoder`, and checksum
`aa4e65ce399b3748`.

## Web Probes

The post-run services started with:

```bash
MODEL_NAME=decoder-40m-3070 docker compose --profile sandbox up --build -d
docker compose --profile web up --build -d web
```

HTTP probes passed for `/v1/models`, `/healthz`, `/api/model`, and `/`.
`/v1/models` disclosed `decoder_accepted_decode_supported=false` and
`degraded_reason="missing decoder route train report"`.

The `/api/chat` probe:

```bash
curl -sS -X POST http://127.0.0.1:8082/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"hello"}'
```

returned a visible non-acceptance outcome:

```json
{
  "stop_reason": "invalid_action",
  "assistant": "",
  "events": [{"kind": "error"}]
}
```

This is not accepted chat quality evidence. It proves the static web and
sandbox path reaches the trained decoder artifact and exposes a concrete
failure stop reason instead of hiding the attempt.

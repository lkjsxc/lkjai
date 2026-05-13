# Native Failure Policy

Owner: `docs/architecture/native/overview/failure-policy.md`.
State: canonical failure policy.

Native failures must be explicit and machine-readable.

## Unsupported Work

Unsupported features return a specific error string instead of falling back to
hidden scaffolds. Current examples:

- Dense and transformer chat decode returns HTTP `422` until the decode
  target lands.
- CPU execution is reported as degraded capability, not as CUDA acceptance.
- Legacy Python `model.pt` checkpoints are not loaded by native serving.

## Artifact Failures

Artifact inspection fails before serving when:

- `manifest.json.format` is not `lkjai-native-artifact`.
- `manifest.json.artifact_kind` is neither `export` nor `checkpoint`.
- required files are absent.
- tensor metadata lacks name, dtype, shape, offset, or byte length.
- tensor ranges exceed the backing binary file.
- manifest checksums do not match config or tokenizer files.

## Runtime Responses

- `/healthz` always returns process status plus artifact and capability state.
- `/v1/models` returns `503` when the model artifact is not loadable.
- `/v1/chat/completions` returns `422` for unsupported dense or transformer
  decode.
- JSON error responses use one `error` string field.

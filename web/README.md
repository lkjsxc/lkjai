# Web

Owner: `web/README.md`.
State: static frontend directory map.

The `web` directory is the static operator UI served by the `web` Compose
profile on port `8080`.

## Files

- [app.js](app.js): browser runtime for status probes, model status display,
  and sandbox `/api/chat` calls.
- [direct.js](direct.js): fallback browser call to inference
  `/v1/chat/completions` when the sandbox agent rejects a non-action reply.
- [index.html](index.html): static chat-first page loaded by nginx.
- [styles.css](styles.css): page layout and visual styling.
- [state.css](state.css): compact status chips and chat-attempt outcome styles.
- [state.js](state.js): small DOM helpers for status chips and attempt errors.
- [nginx.conf](nginx.conf): static file server config. It owns no `/api/*` or
  `/v1/*` routes.

The web profile mounts no model data, source workspace, secrets, or GPU paths.
It talks to the sandbox on `127.0.0.1:8082` and inference on
`127.0.0.1:8081` from the browser.

Open the UI at `http://127.0.0.1:8080` after the sandbox and inference services
are up. A non-acceptance decoder chat attempt is valid when the page shows
assistant text from either the sandbox agent or the direct model fallback. It
also preserves visible `/api/chat` failure stop reasons such as `invalid_action`
or `model_error`. The model panel should still disclose artifact kind, decode
support, degraded reason, and the expected `lkjai_decode_accepted=false` route
detail when it is present in diagnostics.

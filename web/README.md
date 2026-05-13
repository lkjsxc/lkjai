# Web

Owner: `web/README.md`.
State: static frontend directory map.

The `web` directory is the static operator UI served by the `web` Compose
profile on port `8080`.

## Files

- [app.js](app.js): browser runtime for status probes, model status display,
  and sandbox `/api/chat` calls.
- [index.html](index.html): static chat-first page loaded by nginx.
- [styles.css](styles.css): page layout and visual styling.
- [nginx.conf](nginx.conf): static file server config. It owns no `/api/*` or
  `/v1/*` routes.

The web profile mounts no model data, source workspace, secrets, or GPU paths.
It talks to the sandbox on `127.0.0.1:8082` and inference on
`127.0.0.1:8081` from the browser.

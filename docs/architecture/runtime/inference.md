# Model Client Runtime

## Goal

Call a real OpenAI-compatible chat-completions endpoint with live health
verification.

## Contract

- The web runtime must call a real model endpoint; no fake or policy-file
  fallback is allowed in production.
- The runtime must verify the model server is reachable before serving chat
  requests.
- Health probe uses `GET /v1/models` with a 5-second timeout.
- `GET /api/model` returns `reachable` based on the last probe result.

## Request/Response

- Request schema: `model`, `messages`, `max_tokens`, `temperature`.
- Response schema: first `choices[].message.content` is consumed.
- Each model step must return strict JSON action text for the agent parser.

## Failure Semantics

- HTTP request failures surface as explicit transcript error events with stop
  reason `model_error`.
- Non-success status from model server includes status code and body text.
- Parse failures from model response surface as explicit transcript errors.
- No canned fallback answer is allowed when model generation fails.
- If the model server is unreachable, the agent loop stops immediately with
  `model_error`.

## Health Probe Implementation

```rust
async fn health_check(client: &reqwest::Client, url: &str) -> bool {
    let base = url.trim_end_matches("/v1/chat/completions");
    match client.get(format!("{base}/models")).timeout(Duration::from_secs(5)).send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}
```

## Defaults

- `MODEL_API_URL=http://127.0.0.1:8081/v1/chat/completions`
- `MODEL_NAME=qwen3-1.7b-q4`
- `MODEL_MAX_NEW_TOKENS=512`
- `MODEL_TEMPERATURE=0.2`

## Verification

```bash
curl -sf http://127.0.0.1:8081/v1/models | jq '.data[0].id'
curl -sf http://127.0.0.1:8080/api/model | jq '.reachable'
```

Expected: model id is a non-empty string; `reachable` is `true` when the server
is up.

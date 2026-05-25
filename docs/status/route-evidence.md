# Route Evidence

Owner: `docs/status/route-evidence.md`.
State: accepted route evidence map.

## Runtime Transcript

Accepted decode requires a served OpenAI-compatible route transcript at:

```text
data/train/runs/decoder-40m-3070-route-transcript.json
```

The transcript is training evidence because it proves the promoted artifact was
served and decoded through the accepted route after report validation.

Required fields:

- route and request metadata,
- response status and choices,
- decode and KV backend names,
- prefill allocation bytes,
- steady-state token allocation count,
- train-report digest,
- artifact-manifest digest,
- creation timestamp.

## Agent Transcript

Agent chat and tool transcripts are separate runtime records under:

```text
data/agent/runs/
```

They prove sandbox loop behavior, tool execution, and web-visible stop reasons.
They do not promote decoder training or accepted route status.

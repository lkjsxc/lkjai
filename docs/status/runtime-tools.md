# Runtime Tools Status

Owner: `docs/status/runtime-tools.md`.
State: active tool status.

## Profiles

`readonly` exposes:

- `agent.finish`
- `agent.think`
- `fs.read`
- `fs.list`
- `memory.search`
- `resource.search`
- `resource.get`
- `resource.history`

`mutable` adds:

- `agent.request_confirmation`
- `resource.create`
- `resource.update_resource`
- `resource.delete`

`disabled` exposes only `agent.finish`.

## Disabled Tools

These stay rejected in every active profile:

- `memory.write`
- `shell.exec`
- `web.fetch`
- `fs.write`

## Memory

`memory.search` reads deterministic JSONL memory files under
`data/agent/memory/`. Missing memory directories return an empty successful
result. `memory.write` remains disabled until mutation audit coverage exists.

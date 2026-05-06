# Agent Tool Contract

Owner: `docs/product/agent-tools.md`.
State: canonical tool profile and confirmation contract.

## Default Tool Profile

The default runtime profile is `readonly`. It exposes only tools that can run
without changing project, memory, or `kjxlkj` state.

Enabled by default:

- `agent.think`
- `agent.finish`
- `fs.read`
- `fs.list`
- `memory.search`
- `resource.search`
- `resource.fetch`
- `resource.history`
- `resource.preview_markdown`

Disabled in the readonly profile:

- `agent.request_confirmation`
- `resource.create_note`
- `resource.create_media`
- `resource.update_resource`

Disabled in all active profiles until coverage gates pass:

- `shell.exec`
- `web.fetch`
- `fs.write`
- `memory.write`

Set `AGENT_TOOL_PROFILE=mutable` to expose confirmation and confirmed
`kjxlkj` mutation tools. The mutable profile still requires
`agent.request_confirmation` before execution.

## Tool Names

- `shell.exec`: run a command inside the data workspace; disabled by default.
- `web.fetch`: fetch website text; disabled by default.
- `fs.read`: read a workspace file.
- `fs.write`: write a workspace file; disabled by default.
- `fs.list`: list a workspace directory.
- `memory.search`: search durable agent memory.
- `memory.write`: write durable agent memory; disabled by default.
- `resource.search`: search `kjxlkj` resources.
- `resource.fetch`: fetch a `kjxlkj` resource.
- `resource.history`: fetch `kjxlkj` resource history.
- `resource.preview_markdown`: preview a `kjxlkj` markdown mutation.
- `resource.create_note`: create a `kjxlkj` note after confirmation; mutable
  profile only.
- `resource.create_media`: create a `kjxlkj` media resource after confirmation;
  mutable profile only.
- `resource.update_resource`: update a `kjxlkj` resource after confirmation;
  mutable profile only.
- `agent.request_confirmation`: stop and ask before a `kjxlkj` mutation;
  mutable profile only.
- `agent.think`: record a non-terminating visible plan.
- `agent.finish`: terminate successfully with the user-facing answer.

## Selection

- The model selects tools by XML action tags.
- The runtime validates tool names and argument shapes.
- The runtime rejects tools outside `AGENT_TOOL_PROFILE` before execution.
- Slash commands may remain as debug shortcuts.
- Ambiguous natural-language requests are resolved by the model loop.
- The model must call `agent.finish` to return the final answer.
- Everyday chat should normally call `agent.finish` directly without tools.

## YOLO Policy

- Local read-only filesystem and read-only resource tools run without
  confirmation.
- `kjxlkj` mutations require `AGENT_TOOL_PROFILE=mutable`,
  `agent.request_confirmation`, and a later user confirmation before execution.
- Command execution is not sandboxed by the application.
- File and shell tools are bounded to `TOOL_WORKSPACE_DIR`.
- The container must not mount host `/` for agent tools.
- Tool calls must be logged before execution.
- Tool results must be logged after execution.
- Memory writes must be logged.
- The runtime must not invent fake tool results.

## Resource Fields

- `resource.search` accepts `query` or `q`, plus optional `kind`, `sort`,
  `cursor`, `limit`, `direction`, and `scope`.
- `resource.fetch` and `resource.history` accept `ref` or `id`.
- `resource.preview_markdown` accepts `body` and optional
  `current_resource_id`.
- `resource.create_note` accepts `body`, optional `alias`, `is_favorite`, and
  `is_private`.
- `resource.create_media` accepts workspace-confined `path`, optional `alias`,
  `is_favorite`, and `is_private`.
- `resource.update_resource` accepts `ref` or `id`, `body`, optional `alias`,
  `is_favorite`, and `is_private`.

## Limits

- Each tool has a timeout.
- Each textual result is truncated to a configured byte limit.
- Failed tool calls are returned as transcript entries instead of panicking the
  web process.

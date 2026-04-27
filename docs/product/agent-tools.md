# Agent Tool Contract

## Tool Names

- `shell.exec`: run a command inside the data workspace.
- `web.fetch`: fetch website text.
- `fs.read`: read a workspace file.
- `fs.write`: write a workspace file.
- `fs.list`: list a workspace directory.
- `memory.search`: search durable agent memory.
- `memory.write`: write durable agent memory.
- `resource.search`: search `kjxlkj` resources.
- `resource.fetch`: fetch a `kjxlkj` resource.
- `resource.history`: fetch `kjxlkj` resource history.
- `resource.preview_markdown`: preview a `kjxlkj` markdown mutation.
- `resource.create_note`: create a `kjxlkj` note after confirmation.
- `resource.update_resource`: update a `kjxlkj` resource after confirmation.
- `agent.request_confirmation`: stop and ask before a `kjxlkj` mutation.
- `agent.think`: record a non-terminating visible plan.
- `agent.finish`: terminate successfully with the user-facing answer.

## Selection

- The model selects tools by XML action tags.
- The runtime validates tool names and argument shapes.
- Slash commands may remain as debug shortcuts.
- Ambiguous natural-language requests are resolved by the model loop.
- The model must call `agent.finish` to return the final answer.
- Everyday chat should normally call `agent.finish` directly without tools.

## YOLO Policy

- Local filesystem, shell, web, and read-only resource tools run without
  confirmation.
- `kjxlkj` mutations require `agent.request_confirmation`.
- Command execution is not sandboxed by the application.
- File and shell tools are bounded to `TOOL_WORKSPACE_DIR`.
- The container must not mount host `/` for agent tools.
- Tool calls must be logged before execution.
- Tool results must be logged after execution.
- Memory writes must be logged.
- The runtime must not invent fake tool results.

## Limits

- Each tool has a timeout.
- Each textual result is truncated to a configured byte limit.
- Failed tool calls are returned as transcript entries instead of panicking the
  web process.

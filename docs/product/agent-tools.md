# Agent Tool Contract

## Tool Names

- `shell.exec`: run a host command.
- `web.fetch`: fetch website text.
- `fs.read`: read a host file.
- `fs.write`: write a host file.
- `fs.list`: list a host directory.
- `memory.search`: search durable agent memory.
- `memory.write`: write durable agent memory.

## Selection

- The model selects tools by strict JSON action.
- The runtime validates tool names and argument shapes.
- Slash commands may remain as debug shortcuts.
- Ambiguous natural-language requests are resolved by the model loop.

## YOLO Policy

- Tools run without confirmation.
- Command execution is not sandboxed by the application.
- File tools are allowed to access host paths visible inside the container.
- Tool calls must be logged before execution.
- Tool results must be logged after execution.
- Memory writes must be logged.

## Limits

- Each tool has a timeout.
- Each textual result is truncated to a configured byte limit.
- Failed tool calls are returned as transcript entries instead of panicking the
  web process.

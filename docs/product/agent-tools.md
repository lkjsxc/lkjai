# Agent Tool Contract

## Tool Names

- `shell.exec`: run a host command.
- `web.fetch`: fetch website text.
- `file.read`: read a host file.
- `file.write`: write a host file.
- `file.list`: list a host directory.

## YOLO Policy

- Tools run without confirmation.
- Command execution is not sandboxed by the application.
- File tools are allowed to access host paths visible inside the container.
- Tool calls must be logged before execution.
- Tool results must be logged after execution.

## Limits

- Each tool has a timeout.
- Each textual result is truncated to a configured byte limit.
- Failed tool calls are returned as transcript entries instead of panicking the
  web process.

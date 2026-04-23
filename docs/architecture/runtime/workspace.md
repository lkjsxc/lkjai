# Tool Workspace

## Goal

Keep agent-controlled filesystem and command execution inside the mounted data
directory instead of exposing the host root.

## Contract

- `TOOL_WORKSPACE_DIR` defaults to `/app/data/workspace`.
- Compose mounts project `./data` to `/app/data`.
- Compose must not mount host `/` into the web container.
- `shell.exec` runs with current directory set to `TOOL_WORKSPACE_DIR`.
- `fs.read`, `fs.write`, and `fs.list` resolve paths under
  `TOOL_WORKSPACE_DIR`.
- Absolute paths and `..` traversal that escape `TOOL_WORKSPACE_DIR` are
  rejected.
- Tool results are still logged before and after execution.

## Verification

```bash
docker compose --profile verify up --build --abort-on-container-exit verify
```

Expected: tests prove allowed workspace access and blocked root access.

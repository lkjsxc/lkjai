# Tool Workspace

## Goal

Keep agent-controlled filesystem and command execution inside the mounted data
directory instead of exposing the host root.

## Contract

- `TOOL_WORKSPACE_DIR` defaults to `/app/data/workspace`.
- Compose mounts project `./data` to `/app/data`.
- Compose must not mount host `/` into the web container.
- Future `shell.exec` runs with current directory set to `TOOL_WORKSPACE_DIR`.
- `fs.read` and `fs.list` resolve paths under
  `TOOL_WORKSPACE_DIR`.
- `fs.write` is disabled in the default profile.
- Absolute paths and `..` traversal that escape `TOOL_WORKSPACE_DIR` are
  rejected.
- Tool results are still logged before and after execution.

## Verification

```bash
docker compose --progress quiet --profile verify run --build --rm verify
```

Expected: tests prove allowed workspace access and blocked root access.

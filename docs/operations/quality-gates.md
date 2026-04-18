# Quality Gates Contract

## Mandatory Gates

1. `./scripts/check_docs_topology.sh`
2. `./scripts/check_line_limits.sh`
3. `zig fmt --check src/*.zig src/**/*.zig`
4. `zig build`
5. `zig build test`
6. `zig build artifact-size-check`
7. Compose verification bundle in [compose.md](compose.md) (invokes `verify.sh` including `/healthz` and `scripts/verify_api_integration.sh`)

## Stop Rule

- Any failing gate blocks acceptance.

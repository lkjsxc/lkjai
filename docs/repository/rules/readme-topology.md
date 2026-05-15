# README Topology

Owner: `docs/repository/rules/readme-topology.md`.
State: canonical documentation.

## Durable Directories

- Every durable tracked directory has exactly one `README.md`.
- Each `README.md` acts as the local table of contents for LLM agents.
- Durable source directories list immediate source files and child directories.
- Generated shard leaves, caches, and ignored runtime output directories are
  excluded from topology checks.

## Docs Directories

- Every docs directory has at least two children besides `README.md`.
- Parent tables of contents must link immediate children.
- `README.md` files include `Read This Section When` and `Child Index`.

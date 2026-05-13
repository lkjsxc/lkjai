# Repository Rules

## Line Limits

- Docs files stay at `<= 300` lines.
- Authored source files stay at `<= 200` lines.
- Shell, C++, CUDA, CSS, JavaScript, JSON, YAML, TOML, and Markdown source
  files are checked unless explicitly excluded.

## README Topology

- Every durable tracked directory has exactly one `README.md`.
- Each `README.md` acts as the local table of contents for LLM agents.
- Durable source directories list immediate source files and child directories.
- Generated shard leaves, caches, and ignored runtime output directories are
  excluded from topology checks.
- Every docs directory has at least two children besides `README.md`.
- Parent TOCs must link immediate children.

## No Node Rule

- Do not add `package.json`.
- Do not add Node-based verification.
- Do not install Node in Dockerfiles.

## Naming Rule

- Avoid numbered stability labels for repo-owned concepts.
- Literal external route names such as `/v1/models`, external API URLs, and
  runtime fields like `cuda_driver_version` are allowed only where exact names
  are required.
- New local APIs use unnumbered route names.

## Docs Maintenance Checklist

- Update docs before code for contract or behavior changes.
- Keep one `README.md` table of contents per docs directory.
- Keep docs files at `<= 300` lines and source files at `<= 200` lines.
- Link to the owning contract instead of repeating long field lists.
- Require evidence before accepted claims.
- Avoid repo-owned numbered labels outside literal external names.

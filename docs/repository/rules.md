# Repository Rules

## Line Limits

- Docs files stay at `<= 300` lines.
- Authored source files stay at `<= 200` lines.
- Shell, Python, Rust, CSS, JavaScript, TOML, and Markdown source files are
  checked unless explicitly excluded.

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

# Repository Rules

## Line Limits

- Docs files stay at `<= 300` lines.
- Authored source files stay at `<= 200` lines.
- Shell, Python, Rust, CSS, JavaScript, TOML, and Markdown source files are
  checked unless explicitly excluded.

## Docs Topology

- Every docs directory has exactly one `README.md`.
- Every docs directory has at least two children besides `README.md`.
- Parent TOCs must link immediate children.

## No Node Rule

- Do not add `package.json`.
- Do not add Node-based verification.
- Do not install Node in Dockerfiles.

# Line Limits

Owner: `docs/repository/rules/line-limits.md`.
State: canonical documentation.

## Limits

- Docs files stay at `<= 300` lines.
- Authored source files stay at `<= 200` lines.
- Shell, C++, CUDA, CSS, JavaScript, JSON, YAML, TOML, CMake, and Markdown
  source files are checked unless explicitly excluded.

## Splitting Guidance

- Split by behavior or contract owner, not by arbitrary line count.
- Prefer small sibling files under a directory with a local `README.md`.
- Keep generated, ignored, and local evidence out of docs unless summarized.

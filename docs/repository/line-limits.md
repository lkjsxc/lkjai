# Line Limits Contract

## Hard Limits

- Docs files: `<= 300` lines.
- Source files: `<= 200` lines.

## Enforcement

- `scripts/check_line_limits.sh` enforces Markdown and Zig line limits.
- Changes that exceed limits must be split into cohesive modules.

## Guidance

- Prefer narrow single-purpose files.
- Extract helpers instead of compressing readability.


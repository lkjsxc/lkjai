# LLM Optimization Rules

## Formatting Rules

- Use stable section names such as `Goal`, `Contract`, `Defaults`, and
  `Verification`.
- Keep one requirement per bullet.
- Keep canonical definitions in one file and link outward.
- Prefer short declarative statements over narrative paragraphs.
- Delete obsolete contracts instead of preserving conflicting versions.

## Topology Rules

- Every docs directory has exactly one `README.md` table of contents.
- Every docs directory has at least two children besides `README.md`.
- Parent TOCs link immediate children.
- Cross-links are relative.

## Length Rules

- Docs files stay at `<= 300` lines.
- Authored source files stay at `<= 200` lines.
- Edited source files should keep practical headroom below the hard limit.

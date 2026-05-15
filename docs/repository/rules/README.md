# Repository Rules

Owner: `docs/repository/rules/README.md`.
State: canonical documentation.

Use this subtree for authoring constraints that keep the repo readable to LLM
agents and runnable through native tooling.

## Read This Section When

- You are adding or splitting files.
- You are checking README topology or line limits.
- You are choosing names for repo-owned contracts.

## Child Index

- [line-limits.md](line-limits.md): maximum file sizes for docs and source.
- [readme-topology.md](readme-topology.md): README table-of-contents rules.
- [naming.md](naming.md): stable names and discouraged terms.
- [native-only.md](native-only.md): product workflow language and runtime rules.

## Maintenance Checklist

- Update docs before implementation for contract or behavior changes.
- Link to the owning contract instead of repeating long field lists.
- Require evidence before accepted claims.
- Remove conflicting legacy behavior instead of preserving it.

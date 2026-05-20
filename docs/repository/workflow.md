# Workflow

Owner: `docs/repository/workflow.md`.
State: canonical documentation.


## Sequence

1. Update documentation canon.
2. Update implementation to match docs.
3. Run relevant Compose verification.
4. Commit each coherent verified batch.
5. For training changes, verify fixed agent eval artifacts and threshold decisions.

## Commit Policy

- Commit docs-only batches before dependent code batches.
- Prefer small commits with one clear purpose.
- Commit frequently during long-running refactors and training-flow changes.
- Do not accumulate unrelated verified work into one large commit.

## Current Branch Policy

- `main` is the integration target for verified work.
- Feature branches are temporary and are deleted after merge and verification.
- Preserve `tmp/kjxlkj` as untracked reference material.

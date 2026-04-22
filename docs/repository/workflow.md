# Workflow

## Sequence

1. Update documentation canon.
2. Update implementation to match docs.
3. Run relevant Compose verification.
4. Commit each coherent verified batch.
5. For training changes, verify fixed agent eval artifacts and threshold decisions.

## Commit Policy

- Commit docs-only batches before dependent code batches.
- Prefer small commits with one clear purpose.
- Do not accumulate unrelated verified work into one large commit.

## Current Branch Policy

- Work lands on local `main`.
- Preserve `tmp/kjxlkj` as untracked reference material.

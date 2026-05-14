# Artifacts

Owner: `artifacts/README.md`.
State: canonical documentation.


This directory is reserved for generated diagnostics, benchmark logs, plots, raw
telemetry, and patch snapshots produced by the performance tooling.

Generated files under this directory are intentionally ignored by Git. Recreate
them with native Compose workflows:

```bash
docker compose --progress quiet --profile verify run --build --rm verify
docker compose --profile train run --rm train --smoke --steps 2
docker compose --profile train run --rm train --train --mode dense
```

Native benchmark and report CLIs should write machine-readable run directories
here. Keep curated summaries in `docs/operations/performance/`; keep raw run
outputs here.

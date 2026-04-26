# Artifacts

This directory is reserved for generated diagnostics, benchmark logs, plots, raw
telemetry, and patch snapshots produced by the performance tooling.

Generated files under this directory are intentionally ignored by Git. Recreate
them with:

```bash
python3 diagnostics/collect.py --run-id final --timeout 15
python3 benchmarks/run_matrix.py --run-id smoke --cases real_legacy,real_mapped,synthetic_gpu --repeats 1
python3 reports/generate_report.py --diagnostics-run-id final
```

Keep durable summaries in `reports/`; keep raw run outputs here.

# Performance

This subtree is the canonical performance contract for `lkjai`.

## Read This Section When

- You need benchmark, profiling, or validation rules.
- You need performance report and promotion contracts.
- You need accepted evidence or hardware profile policy.

## Purpose

- Maximize training throughput for the active `scratch-40m` model contract.
- Keep optimization decisions measurable and reproducible.
- Prefer repo-local, Compose-runnable workflows over one-off host commands.

## Child Index

- [contracts/README.md](contracts/README.md): report fields, benchmark outputs,
  artifacts, and promotion criteria.
- [measurement/README.md](measurement/README.md): benchmark protocol,
  profiling, validation, training speed, and kernel planning.
- [profiles/README.md](profiles/README.md): RTX 3070 acceptance and larger
  profile planning.
- [evidence/README.md](evidence/README.md): dated dense and decoder evidence
  records.

## Active Priority

Training speed is the first priority. Inference improvements are accepted when
they share the same model/cache foundations or remove obvious decode waste.

## Route By Owner

- Report field changes: [contracts/train-report-fields.md](contracts/train-report-fields.md).
- Benchmark-generated files: [contracts/benchmark-artifacts.md](contracts/benchmark-artifacts.md).
- Promotion and rejection logic: [contracts/promotion-criteria.md](contracts/promotion-criteria.md).
- Measurement protocol: [measurement/benchmarking.md](measurement/benchmarking.md).
- Accepted evidence records: [evidence/evidence.md](evidence/evidence.md).

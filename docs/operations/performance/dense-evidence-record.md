# Dense Evidence Record

Use this format when documenting a dense CUDA throughput change. Generated raw
outputs stay under ignored `artifacts/` and `data/perf-runs/`; committed docs
contain only curated evidence.

## Required Fields

- Commit range and short summary.
- GPU name, compute capability, memory, driver, CUDA runtime, and cuDNN version.
- Native config path, training config path, batch size, sequence length,
  gradient accumulation, optimizer steps, and run purpose.
- Exact command used to build or run.
- Artifact paths for copied train report, summary JSON, and comparison output.
- Before and after `tokens_per_second`, step seconds, H2D, forward, backward,
  optimizer, checkpoint, and export timings.
- Dense transient bytes: logits, grad logits, hidden gradient, logits readback,
  and cuBLASLt workspace.
- Tuning fields: autotune mode, workspace sweep, selected algo ordinals,
  allocator backend, workspace high water, reallocations, timing mode, and
  LM-head FP32 cache refreshes.
- Correctness fields: checkpoint checksum, export checksum, logits checksum,
  logits reference status, max absolute difference, and tolerance.
- Limitations and non-goals.

## Method

Use matched reports for performance claims. A matched pair has the same config
digest, packed-cache digest, batch size, sequence length, gradient
accumulation, optimizer-step count, CUDA architecture flags, and GPU profile.

Promotion evidence must come from accepted dense reports. Transformer,
unsupported-decode, and 40M diagnostic-only records may be listed as
diagnostics, but they must not be described as accepted dense promotion.

## Blog Notes

When preserving a useful engineering narrative, record:

- what changed,
- why the old path was limiting throughput,
- which report fields prove the new path was active,
- what benchmark was used,
- what improved or regressed,
- which constraints remain for RTX 3070 and newer GPUs.

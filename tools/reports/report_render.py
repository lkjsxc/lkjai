import json

from report_helpers import PRIMARY_SOURCES, REPORT_ROOT, ROOT, fmt, latest_dir


def render_report(context: dict) -> None:
    source_lines = "\n".join(f"- [{name}]({url})" for name, url in PRIMARY_SOURCES)
    rust_summary = rust_pilot_summary()
    out_dir = REPORT_ROOT / str(context.get("run_id", "latest"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "training-performance-report.md"
    report.write_text(
        f"""# lkjai Training Performance Report

## Executive Summary

The implemented suite records environment diagnostics, reproducible training benchmarks, telemetry, charts, diffs, and a Rust pilot path. The current host is WSL2, so native Linux observability is reduced for PCI root-port inspection, perf events, and some NVIDIA management operations.

Best known configuration: {context['best']}

## Environment And Tooling

Diagnostics directory: `{context['diagnostics_label']}`

{context['tool_table']}

Known hardware facts from discovery: RTX 3070, 8 GiB VRAM, compute capability 8.6, PCIe Gen4 x16 observed, NVLink unsupported, WDDM/WSL driver path.

## Training Pipeline

```mermaid
flowchart LR
  A[JSONL corpus shards] --> B[Byte BPE tokenizer]
  B --> C[Packed token cache]
  C --> D[DataLoader or mapped reader]
  D --> E[ScratchLM forward/backward]
  E --> F[AdamW optimizer]
  F --> G[Validation and checkpoints]
```

```mermaid
timeline
  title Optimization Timeline
  Baseline discovery : entrypoint, model, data, GPU, WSL limits
  Harness : diagnostics collector, benchmark matrix, telemetry
  Data path : legacy file-open reader vs mapped packed cache
  GPU path : AMP, compile, TF32, batch and accumulation sweeps
  Rust path : packed-reader pilot and migration plan
  Report : tables, charts, diffs, best known config
```

## Benchmark Results

Benchmark directory: `{context['benchmark_label']}`

{context['aggregate_table']}

### Before/After Comparisons

Baseline for this table is the repeated `matrix-short/real_legacy` case. These are short synchronized microbenchmarks, so use them to rank candidates before longer confirmation runs.

{context['comparison_table']}

### Per-Run Details

{context['details_table']}

Charts:

{context['chart_links']}

## Bottleneck Ranking

1. GPU/model execution is the primary short-run bottleneck: real data and synthetic GPU data were close, so the input path is not dominating these measured steps.
2. `torch.compile` is the strongest post-warm candidate, but first-step compilation is expensive and requires the compile-ready image with `gcc/g++`.
3. Activation checkpointing is expensive in the short probe; disabling it improved step time, but longer VRAM and loss checks are required before accepting it for full runs.
4. Increasing batch size improves tokens/sec up to batch 2 in the short probe; batch 4 fit but was slower, so the current knee candidate is batch 2.
5. The mapped loader is implemented and useful for Rust/Python parity work, but it did not beat the legacy loader in the repeated short GPU-bound matrix.
6. WSL2 limits profiler and PCI inspection depth; `nvidia-smi pci -gCnt`, dmon/pmon, and query loops are the reliable fallbacks on this host.

## Implemented Changes

- Added env-gated training controls for data mode, loader implementation, DataLoader workers, pinning, prefetch, persistent workers, compile mode, TF32, matmul precision, clipping, and profiling.
- Added `MappedPackedDataset` to avoid repeated per-sample file opens and Python list conversion.
- Added synthetic CPU and synthetic GPU modes for the required real-vs-synthetic diagnostic decision.
- Added step profiling JSONL with loader wait, H2D, forward, backward, optimizer, loss, and token counts.
- Added diagnostics collector, Docker benchmark matrix, report generator, and Rust pilot scaffold.

## Correctness Checks

- Docker Python tests: `60 passed, 2 deselected`.
- Benchmark runs used `TRAIN_RESUME=never`, fixed seeds, isolated data directories, finite losses, and no checkpoint overwrite of the existing `data/train` artifact.
- Existing full artifact quality remains a separate concern: prior behavioral eval was 0% with mostly invalid XML, so throughput improvements do not imply model competency improvement.

Diff artifact: `{context['diff_path']}`

## Rust Migration Plan

{rust_summary or 'Rust pilot benchmark was not available when this report was generated.'}

Recommended order:

1. Packed cache reading and collation: memory-map `tokens.bin`, `loss_mask.bin`, and `starts.bin`; expose batch windows through PyO3/maturin if Python mapped tensors are still costly.
2. JSONL manifest and source fingerprinting: Rust can reduce Python parsing overhead during cache rebuilds.
3. CPU preprocessing and text normalization: port only if diagnostics show tokenizer/pre-cache work dominates.
4. Full Rust training loop: defer. PyTorch CUDA kernels and optimizer behavior dominate training semantics; `tch-rs` or Burn would increase maintenance risk unless profiling proves Python loop overhead is the main limiter.

## Assumptions And Blocked Items

- Existing dirty corpus data is preserved.
- Clock and power-limit experiments are reversible but currently blocked by insufficient permission in WSL.
- `nvidia-smi stats` is unavailable here; query loops, dmon, pmon, and PCI counters are used instead.
- Nsight Systems, Nsight Compute, perf, fio, iostat, pidstat, iotop, lspci, rustc, and cargo may be missing on the host; diagnostics record exact availability.

## Primary Sources

{source_lines}
""",
        encoding="utf-8",
    )
    context["report_path"] = str(report.relative_to(ROOT))


def rust_pilot_summary() -> str:
    rust_dir = latest_dir(ROOT / "artifacts" / "experiments")
    if not rust_dir:
        return ""
    rust_path = rust_dir / "rust-packed-reader.json"
    py_path = rust_dir / "python-packed-reader.json"
    if not rust_path.exists() or not py_path.exists():
        return ""
    try:
        rust_json = parse_embedded_json(rust_path)
        py_json = parse_embedded_json(py_path)
        speedup = float(py_json["elapsed_seconds"]) / max(1e-9, float(rust_json["elapsed_seconds"]))
        return f"Rust pilot: {fmt(rust_json['elapsed_seconds'], 6)} s for {rust_json['windows_read']} windows vs Python {fmt(py_json['elapsed_seconds'], 4)} s ({fmt(speedup, 1)}x faster for the read/collate microbenchmark)."
    except (ValueError, KeyError):
        return f"Rust pilot artifacts are present in `{rust_dir.relative_to(ROOT)}`."


def parse_embedded_json(path):
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.find("{") : text.rfind("}") + 1])

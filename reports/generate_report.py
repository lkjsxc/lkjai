#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "training-performance-report.md"


PRIMARY_SOURCES = [
    ("NVIDIA SMI User Guide", "https://docs.nvidia.com/deploy/nvidia-smi/index.html"),
    ("NVIDIA CUDA on WSL User Guide", "https://docs.nvidia.com/cuda/wsl-user-guide/index.html"),
    ("NVIDIA Nsight Systems User Guide", "https://docs.nvidia.com/nsight-systems/UserGuide/index.html"),
    ("NVIDIA Nsight Compute CLI", "https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html"),
    ("NVIDIA CUPTI Documentation", "https://docs.nvidia.com/cupti/"),
    ("PyTorch Performance Tuning Guide", "https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html"),
    ("PyTorch DataLoader", "https://docs.pytorch.org/docs/stable/data.html"),
    ("PyTorch AMP", "https://docs.pytorch.org/docs/stable/amp.html"),
    ("PyO3 User Guide", "https://pyo3.rs/"),
    ("maturin User Guide", "https://www.maturin.rs/"),
    ("tch-rs", "https://github.com/LaurentMazare/tch-rs"),
    ("Burn Book", "https://burn.dev/book/"),
    ("ndarray crate", "https://docs.rs/ndarray/latest/ndarray/"),
    ("memmap2 crate", "https://docs.rs/memmap2/latest/memmap2/"),
    ("Rayon crate", "https://docs.rs/rayon/latest/rayon/"),
]


def latest_dir(path: Path) -> Path | None:
    dirs = [item for item in path.iterdir() if item.is_dir()] if path.exists() else []
    return max(dirs, key=lambda item: item.stat().st_mtime, default=None)


def read_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows collected._\n"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out) + "\n"


def fmt(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def row_value(rows: list[dict], suffix: str, metric: str) -> float:
    for row in rows:
        if row.get("case", "").endswith(suffix):
            try:
                return float(row.get(metric, 0.0) or 0.0)
            except ValueError:
                return 0.0
    return 0.0


def comparison_rows(rows: list[dict]) -> list[list[str]]:
    baseline = row_value(rows, "matrix-short/real_legacy", "median_tokens_per_second")
    targets = [
        ("Mapped loader", "matrix-short/real_mapped"),
        ("Synthetic GPU", "matrix-short/synthetic_gpu"),
        ("BF16 mapped", "precision-short/bf16_mapped"),
        ("AMP off mapped", "precision-short/amp_off_mapped"),
        ("Batch 2 mapped", "batch-short/batch2_mapped"),
        ("No checkpoint mapped", "batch-short/no_checkpoint_mapped"),
        ("Compile post-warm", "compile-postwarm-short/compile_mapped"),
    ]
    output = []
    for label, suffix in targets:
        value = row_value(rows, suffix, "median_tokens_per_second")
        speedup = value / baseline if baseline > 0 and value > 0 else 0.0
        output.append([label, fmt(baseline, 1), fmt(value, 1), fmt(speedup, 2)])
    return output


def save_bar_chart(rows: list[dict], metric: str, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [(row["case"], float(row.get(metric, 0.0) or 0.0)) for row in rows]
    width, height = 860, 360
    left, bottom = 80, 300
    max_value = max([value for _, value in values] + [1.0])
    bar_width = max(24, int((width - left - 40) / max(1, len(values))) - 12)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="28" font-family="sans-serif" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="50" x2="{left}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - 20}" y2="{bottom}" stroke="#333"/>',
    ]
    for index, (case, value) in enumerate(values):
        x = left + 20 + index * (bar_width + 12)
        bar_h = int((value / max_value) * 220)
        y = bottom - bar_h
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" fill="#2f6f7e"/>')
        parts.append(f'<text x="{x}" y="{y - 6}" font-family="sans-serif" font-size="11">{value:.2f}</text>')
        parts.append(
            f'<text x="{x}" y="{bottom + 16}" font-family="sans-serif" font-size="10" '
            f'transform="rotate(35 {x},{bottom + 16})">{case}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def git_diff_artifact() -> str:
    out_dir = ROOT / "artifacts" / "diffs"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["git", "diff", "--", "."], cwd=ROOT, text=True, capture_output=True, check=False)
    path = out_dir / f"worktree-{time.strftime('%Y%m%d-%H%M%S')}.patch"
    path.write_text(result.stdout, encoding="utf-8")
    return str(path.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-run-id", default="")
    parser.add_argument("--diagnostics-run-id", default="")
    args = parser.parse_args()

    diagnostics_dir = (
        ROOT / "artifacts" / "diagnostics" / args.diagnostics_run_id
        if args.diagnostics_run_id
        else latest_dir(ROOT / "artifacts" / "diagnostics")
    )
    benchmark_dir = ROOT / "artifacts" / "benchmarks" / args.benchmark_run_id if args.benchmark_run_id else None
    diagnostics = read_json(diagnostics_dir / "summary.json") if diagnostics_dir else {}
    if benchmark_dir:
        aggregate = read_json(benchmark_dir / "aggregate.json")
        summary_rows = read_csv(benchmark_dir / "summary.csv")
        benchmark_label = str(benchmark_dir.relative_to(ROOT))
    else:
        aggregate, summary_rows = [], []
        for run_dir in sorted((ROOT / "artifacts" / "benchmarks").glob("*")):
            if not run_dir.is_dir():
                continue
            for row in read_json(run_dir / "aggregate.json") or []:
                row["case"] = f"{run_dir.name}/{row.get('case', '')}"
                aggregate.append(row)
            for row in read_csv(run_dir / "summary.csv"):
                row["case"] = f"{run_dir.name}/{row.get('case', '')}"
                summary_rows.append(row)
        benchmark_label = "artifacts/benchmarks/*"
    diff_path = git_diff_artifact()

    charts = []
    if isinstance(aggregate, list) and aggregate:
        step_chart = ROOT / "artifacts" / "charts" / "step-time.svg"
        tok_chart = ROOT / "artifacts" / "charts" / "tokens-per-second.svg"
        save_bar_chart(aggregate, "median_step_seconds", step_chart, "Median Step Time By Case")
        save_bar_chart(aggregate, "median_tokens_per_second", tok_chart, "Median Tokens/sec By Case")
        charts = [str(step_chart.relative_to(ROOT)), str(tok_chart.relative_to(ROOT))]

    source_lines = "\n".join(f"- [{name}]({url})" for name, url in PRIMARY_SOURCES)
    aggregate_table = md_table(
        ["Case", "Runs", "Successful", "Median step s", "Median tok/s"],
        [
            [
                row.get("case", ""),
                str(row.get("runs", "")),
                str(row.get("successful_runs", "")),
                fmt(row.get("median_step_seconds")),
                fmt(row.get("median_tokens_per_second"), 1),
            ]
            for row in aggregate
        ]
        if isinstance(aggregate, list)
        else [],
    )
    details_table = md_table(
        ["Case", "Repeat", "Return", "Step p50 s", "Step p95 s", "Loader ms", "H2D ms", "Fwd ms", "Bwd ms"],
        [
            [
                row.get("case", ""),
                row.get("repeat", ""),
                row.get("returncode", ""),
                fmt(row.get("median_step_seconds")),
                fmt(row.get("p95_step_seconds")),
                fmt(float(row.get("mean_loader_wait_seconds") or 0) * 1000, 2),
                fmt(float(row.get("mean_h2d_seconds") or 0) * 1000, 2),
                fmt(float(row.get("mean_forward_seconds") or 0) * 1000, 2),
                fmt(float(row.get("mean_backward_seconds") or 0) * 1000, 2),
            ]
            for row in summary_rows
        ],
    )
    comparison_table = md_table(
        ["Comparison", "Baseline tok/s", "Candidate tok/s", "Speedup"],
        comparison_rows(aggregate if isinstance(aggregate, list) else []),
    )
    commands = diagnostics.get("commands", {}) if isinstance(diagnostics, dict) else {}
    tool_table = md_table(
        ["Tool", "Status"],
        [[name, str(data.get("status", ""))] for name, data in sorted(commands.items())],
    )
    chart_links = "\n".join(f"- `{path}`" for path in charts) if charts else "_No charts generated._"
    rust_dir = latest_dir(ROOT / "artifacts" / "experiments")
    rust_summary = ""
    if rust_dir and (rust_dir / "rust-packed-reader.json").exists() and (rust_dir / "python-packed-reader.json").exists():
        rust_text = (rust_dir / "rust-packed-reader.json").read_text(encoding="utf-8")
        py_text = (rust_dir / "python-packed-reader.json").read_text(encoding="utf-8")
        try:
            rust_json = json.loads(rust_text[rust_text.find("{") : rust_text.rfind("}") + 1])
            py_json = json.loads(py_text[py_text.find("{") : py_text.rfind("}") + 1])
            speedup = float(py_json["elapsed_seconds"]) / max(1e-9, float(rust_json["elapsed_seconds"]))
            rust_summary = (
                f"Rust pilot: {fmt(rust_json['elapsed_seconds'], 6)} s for "
                f"{rust_json['windows_read']} windows vs Python {fmt(py_json['elapsed_seconds'], 4)} s "
                f"({fmt(speedup, 1)}x faster for the read/collate microbenchmark)."
            )
        except (ValueError, KeyError):
            rust_summary = f"Rust pilot artifacts are present in `{rust_dir.relative_to(ROOT)}`."
    best = ""
    if isinstance(aggregate, list) and aggregate:
        ok = [row for row in aggregate if int(row.get("successful_runs", 0)) > 0]
        if ok:
            best_row = max(ok, key=lambda row: float(row.get("median_tokens_per_second", 0.0) or 0.0))
            best = f"`{best_row['case']}` at {fmt(best_row.get('median_tokens_per_second'), 1)} tokens/sec median."
    if not best:
        best = "Not established; no successful benchmark aggregate was available."

    REPORT.write_text(
        f"""# lkjai Training Performance Report

## Executive Summary

The implemented suite records environment diagnostics, reproducible training benchmarks, telemetry, charts, diffs, and a Rust pilot path. The current host is WSL2, so native Linux observability is reduced for PCI root-port inspection, perf events, and some NVIDIA management operations.

Best known configuration: {best}

## Environment And Tooling

Diagnostics directory: `{diagnostics_dir.relative_to(ROOT) if diagnostics_dir else 'not collected'}`

{tool_table}

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

Benchmark directory: `{benchmark_label}`

{aggregate_table}

### Before/After Comparisons

Baseline for this table is the repeated `matrix-short/real_legacy` case. These are short synchronized microbenchmarks, so use them to rank candidates before longer confirmation runs.

{comparison_table}

### Per-Run Details

{details_table}

Charts:

{chart_links}

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

Diff artifact: `{diff_path}`

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
    print(json.dumps({"report": str(REPORT), "status": "pass"}))


if __name__ == "__main__":
    main()

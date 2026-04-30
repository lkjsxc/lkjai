#!/usr/bin/env python3
import argparse
import json

from report_helpers import (
    ROOT,
    comparison_rows,
    fmt,
    git_diff_artifact,
    latest_dir,
    md_table,
    read_csv,
    read_json,
    save_bar_chart,
)
from report_render import render_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-run-id", default="")
    parser.add_argument("--diagnostics-run-id", default="")
    args = parser.parse_args()
    diagnostics_dir = selected_dir(ROOT / "artifacts" / "diagnostics", args.diagnostics_run_id)
    aggregate, summary_rows, benchmark_label = benchmark_data(args.benchmark_run_id)
    charts = chart_links(aggregate)
    context = {
        "run_id": args.benchmark_run_id or args.diagnostics_run_id or "latest",
        "best": best_config(aggregate),
        "diagnostics_label": diagnostics_dir.relative_to(ROOT) if diagnostics_dir else "not collected",
        "benchmark_label": benchmark_label,
        "tool_table": tool_table(read_json(diagnostics_dir / "summary.json") if diagnostics_dir else {}),
        "aggregate_table": aggregate_table(aggregate),
        "comparison_table": comparison_table(aggregate),
        "details_table": details_table(summary_rows),
        "chart_links": "\n".join(f"- `{path}`" for path in charts) if charts else "_No charts generated._",
        "diff_path": git_diff_artifact(),
    }
    render_report(context)
    print(json.dumps({"report": context["report_path"], "status": "pass"}))


def selected_dir(root, run_id: str):
    return root / run_id if run_id else latest_dir(root)


def benchmark_data(run_id: str):
    if run_id:
        root = ROOT / "artifacts" / "benchmarks" / run_id
        return read_json(root / "aggregate.json"), read_csv(root / "summary.csv"), str(root.relative_to(ROOT))
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
    return aggregate, summary_rows, "artifacts/benchmarks/*"


def chart_links(aggregate) -> list[str]:
    if not isinstance(aggregate, list) or not aggregate:
        return []
    step_chart = ROOT / "artifacts" / "charts" / "step-time.svg"
    tok_chart = ROOT / "artifacts" / "charts" / "tokens-per-second.svg"
    save_bar_chart(aggregate, "median_step_seconds", step_chart, "Median Step Time By Case")
    save_bar_chart(aggregate, "median_tokens_per_second", tok_chart, "Median Tokens/sec By Case")
    return [str(step_chart.relative_to(ROOT)), str(tok_chart.relative_to(ROOT))]


def aggregate_table(aggregate) -> str:
    rows = [
        [row.get("case", ""), str(row.get("runs", "")), str(row.get("successful_runs", "")), fmt(row.get("median_step_seconds")), fmt(row.get("median_tokens_per_second"), 1)]
        for row in aggregate
    ] if isinstance(aggregate, list) else []
    return md_table(["Case", "Runs", "Successful", "Median step s", "Median tok/s"], rows)


def details_table(summary_rows) -> str:
    rows = []
    for row in summary_rows:
        rows.append([
            row.get("case", ""),
            row.get("repeat", ""),
            row.get("returncode", ""),
            fmt(row.get("median_step_seconds")),
            fmt(row.get("p95_step_seconds")),
            fmt(float(row.get("mean_loader_wait_seconds") or 0) * 1000, 2),
            fmt(float(row.get("mean_h2d_seconds") or 0) * 1000, 2),
            fmt(float(row.get("mean_forward_seconds") or 0) * 1000, 2),
            fmt(float(row.get("mean_backward_seconds") or 0) * 1000, 2),
        ])
    headers = ["Case", "Repeat", "Return", "Step p50 s", "Step p95 s", "Loader ms", "H2D ms", "Fwd ms", "Bwd ms"]
    return md_table(headers, rows)


def comparison_table(aggregate) -> str:
    return md_table(["Comparison", "Baseline tok/s", "Candidate tok/s", "Speedup"], comparison_rows(aggregate if isinstance(aggregate, list) else []))


def tool_table(diagnostics) -> str:
    commands = diagnostics.get("commands", {}) if isinstance(diagnostics, dict) else {}
    return md_table(["Tool", "Status"], [[name, str(data.get("status", ""))] for name, data in sorted(commands.items())])


def best_config(aggregate) -> str:
    if isinstance(aggregate, list) and aggregate:
        ok = [row for row in aggregate if int(row.get("successful_runs", 0)) > 0]
        if ok:
            row = max(ok, key=lambda item: float(item.get("median_tokens_per_second", 0.0) or 0.0))
            return f"`{row['case']}` at {fmt(row.get('median_tokens_per_second'), 1)} tokens/sec median."
    return "Not established; no successful benchmark aggregate was available."


if __name__ == "__main__":
    main()

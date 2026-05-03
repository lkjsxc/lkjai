import json
import math
from pathlib import Path

from benchmark_reports import summarize_train_report


def phase_summary(
    report: dict,
    wall_elapsed: float,
    code: int,
    data_dir: Path,
    model_name: str,
) -> dict:
    summary = summarize_train_report(report)
    summary.update(
        {
            "returncode": code,
            "wall_elapsed_seconds": wall_elapsed,
            "config_digest": report.get("config_digest", ""),
            "dataset_digest": report.get("dataset_digest", ""),
            "limitations": report.get("limitations", []),
            "artifact_paths": {
                "data_dir": str(data_dir),
                "checkpoint": str(data_dir / "checkpoints" / "latest"),
                "export": str(data_dir / "exports" / model_name),
            },
        }
    )
    return summary


def target_steps_from_pilot(report: dict, target_seconds: int) -> int:
    steps = int(report.get("optimizer_steps", report.get("steps", 0)))
    elapsed = float(report.get("elapsed_seconds", 0.0))
    if steps <= 0 or elapsed <= 0 or not math.isfinite(elapsed):
        raise SystemExit("pilot did not produce usable elapsed/step calibration")
    return max(1, int(target_seconds / (elapsed / steps)))


def train_blocker(phase: str, code: int, report: dict) -> str:
    status = report.get("status", "")
    if code == 3 and status == "success":
        change = report.get("weight_change", {})
        reason = change.get("reason") or "native tensor-delta check failed"
        return (
            f"{phase} training returned code 3 from the native weight-change "
            f"gate: {reason}"
        )
    return f"{phase} training failed with return code {code}"


def write_markdown(summary: dict, path: Path) -> None:
    pilot = summary.get("pilot", {})
    full = summary.get("full", {})
    paths = full.get("artifact_paths", pilot.get("artifact_paths", {}))
    lines = [
        "# Dense BF16 CUDA Training Run",
        "",
        f"- run id: `{summary['run_id']}`",
        f"- status: `{summary['runner_status']}`",
        f"- full status: `{summary.get('full_status', '')}`",
        f"- GPU: `{pilot.get('cuda_device_name', '')}`",
        f"- CUDA runtime: `{pilot.get('cuda_runtime_version', 0)}`",
        f"- driver: `{pilot.get('cuda_driver_version', 0)}`",
        f"- arch flags: `{pilot.get('cuda_arch_flags', '')}`",
        f"- config digest: `{pilot.get('config_digest', '')}`",
        f"- cache digest: `{summary.get('cache', {}).get('packed_data_checksum', '')}`",
        f"- pilot tokens/sec: `{pilot.get('median_tokens_per_second', 0.0)}`",
        f"- pilot optimizer steps: `{pilot.get('optimizer_steps', 0)}`",
        f"- pilot weight-change status: `{pilot.get('weight_change_status', '')}`",
        f"- calibrated full steps: `{summary.get('target_optimizer_steps', 0)}`",
        f"- full tokens/sec: `{full.get('median_tokens_per_second', 0.0)}`",
        f"- full optimizer steps: `{full.get('optimizer_steps', 0)}`",
        f"- full weight-change status: `{full.get('weight_change_status', '')}`",
        f"- checkpoint: `{paths.get('checkpoint', '')}`",
        f"- export: `{paths.get('export', '')}`",
        "",
        "## Limitations",
        "",
    ]
    limitations = summary.get("limitations", [])
    if limitations:
        lines.extend(f"- `{item}`" for item in limitations)
    else:
        lines.append("- none recorded")
    if summary.get("blocker"):
        lines.extend(["", "## Blocker", "", summary["blocker"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

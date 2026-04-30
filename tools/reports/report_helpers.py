import csv
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "artifacts" / "reports"
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
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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
        ("Batch-mapped loader", "matrix-short/real_batch_mapped"),
        ("Synthetic GPU", "matrix-short/synthetic_gpu"),
        ("BF16 batch-mapped", "precision-short/bf16_batch_mapped"),
        ("AMP off batch-mapped", "precision-short/amp_off_batch_mapped"),
        ("Batch 2 batch-mapped", "batch-short/batch2_batch_mapped"),
        ("No checkpoint batch-mapped", "batch-short/no_checkpoint_batch_mapped"),
        ("Compile post-warm", "compile-postwarm-short/compile_batch_mapped"),
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
    width, height, left, bottom = 860, 360, 80, 300
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
        parts.append(f'<text x="{x}" y="{bottom + 16}" font-family="sans-serif" font-size="10" transform="rotate(35 {x},{bottom + 16})">{case}</text>')
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")


def git_diff_artifact() -> str:
    out_dir = ROOT / "artifacts" / "diffs"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["git", "diff", "--", "."], cwd=ROOT, text=True, capture_output=True, check=False)
    path = out_dir / f"worktree-{time.strftime('%Y%m%d-%H%M%S')}.patch"
    path.write_text(result.stdout, encoding="utf-8")
    return str(path.relative_to(ROOT))

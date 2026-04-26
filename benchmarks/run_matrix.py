#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GPU_QUERY = (
    "timestamp,index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.free,"
    "clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu,"
    "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"
)


CASES = {
    "real_legacy": {"TRAIN_DATA_MODE": "real", "TRAIN_DATALOADER_IMPL": "legacy"},
    "real_mapped": {"TRAIN_DATA_MODE": "real", "TRAIN_DATALOADER_IMPL": "mapped"},
    "synthetic_cpu": {"TRAIN_DATA_MODE": "synthetic_cpu", "TRAIN_DATALOADER_IMPL": "legacy"},
    "synthetic_gpu": {"TRAIN_DATA_MODE": "synthetic_gpu", "TRAIN_DATALOADER_IMPL": "legacy"},
    "bf16_mapped": {"TRAIN_DATA_MODE": "real", "TRAIN_DATALOADER_IMPL": "mapped", "TRAIN_AMP": "bf16"},
    "fp16_mapped": {"TRAIN_DATA_MODE": "real", "TRAIN_DATALOADER_IMPL": "mapped", "TRAIN_AMP": "fp16"},
    "amp_off_mapped": {"TRAIN_DATA_MODE": "real", "TRAIN_DATALOADER_IMPL": "mapped", "TRAIN_AMP": "off"},
    "compile_mapped": {
        "TRAIN_DATA_MODE": "real",
        "TRAIN_DATALOADER_IMPL": "mapped",
        "TRAIN_TORCH_COMPILE": "1",
        "TRAIN_TORCH_COMPILE_MODE": "reduce-overhead",
    },
    "batch2_mapped": {"TRAIN_DATA_MODE": "real", "TRAIN_DATALOADER_IMPL": "mapped", "TRAIN_BATCH_SIZE": "2"},
    "batch4_mapped": {"TRAIN_DATA_MODE": "real", "TRAIN_DATALOADER_IMPL": "mapped", "TRAIN_BATCH_SIZE": "4"},
    "no_checkpoint_mapped": {
        "TRAIN_DATA_MODE": "real",
        "TRAIN_DATALOADER_IMPL": "mapped",
        "TRAIN_GRADIENT_CHECKPOINTING": "0",
    },
}


def run(command: list[str], log_path: Path | None = None, env: dict | None = None) -> int:
    with subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as process:
        lines = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            lines.append(line)
        code = process.wait()
    if log_path:
        log_path.write_text("".join(lines), encoding="utf-8")
    return code


def copy_or_link_tree(src: Path, dst: Path) -> None:
    if not src.exists() or dst.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)


def prepare_data_dir(data_dir: Path) -> None:
    source = ROOT / "data" / "train"
    data_dir.mkdir(parents=True, exist_ok=True)
    copy_or_link_tree(source / "tokenizer", data_dir / "tokenizer")
    copy_or_link_tree(source / "datasets" / "packed", data_dir / "datasets" / "packed")


def read_proc_stat() -> tuple[list[int], list[int]]:
    totals, idles = [], []
    for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
        if not line.startswith("cpu"):
            continue
        name, *items = line.split()
        if name == "cpu":
            continue
        values = [int(item) for item in items]
        idle = values[3] + values[4]
        totals.append(sum(values))
        idles.append(idle)
    return totals, idles


def cpu_percent(prev: tuple[list[int], list[int]], curr: tuple[list[int], list[int]]) -> list[float]:
    output = []
    for total_a, idle_a, total_b, idle_b in zip(prev[0], prev[1], curr[0], curr[1]):
        total_delta = max(1, total_b - total_a)
        idle_delta = idle_b - idle_a
        output.append(100.0 * (1.0 - idle_delta / total_delta))
    return output


class Telemetry:
    def __init__(self, out_dir: Path, interval: float):
        self.out_dir = out_dir
        self.interval = interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        gpu_path = self.out_dir / "gpu.csv"
        cpu_path = self.out_dir / "cpu.csv"
        mem_path = self.out_dir / "memory.csv"
        pci_path = self.out_dir / "pcie-counters.txt"
        prev = read_proc_stat()
        with gpu_path.open("w", encoding="utf-8") as gpu_file, cpu_path.open("w", encoding="utf-8") as cpu_file:
            cpu_writer = csv.writer(cpu_file)
            cpu_writer.writerow(["timestamp", "mean_cpu", "max_cpu", "per_core"])
            while not self.stop.is_set():
                timestamp = time.time()
                self._append_gpu(gpu_file)
                curr = read_proc_stat()
                values = cpu_percent(prev, curr)
                prev = curr
                cpu_writer.writerow([timestamp, statistics.fmean(values), max(values), " ".join(f"{v:.2f}" for v in values)])
                cpu_file.flush()
                mem_path.write_text(Path("/proc/meminfo").read_text(encoding="utf-8"), encoding="utf-8")
                pci = subprocess.run(["nvidia-smi", "pci", "-gCnt"], text=True, capture_output=True, check=False)
                pci_path.write_text(pci.stdout + pci.stderr, encoding="utf-8")
                self.stop.wait(self.interval)

    def _append_gpu(self, handle) -> None:
        command = ["nvidia-smi", f"--query-gpu={DEFAULT_GPU_QUERY}", "--format=csv"]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            handle.write(result.stderr)
            handle.flush()
            return
        lines = result.stdout.splitlines()
        if handle.tell() == 0:
            handle.write(result.stdout)
        elif len(lines) > 1:
            handle.write(lines[-1] + "\n")
        handle.flush()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def summarize_steps(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    step_times = [item["microstep_seconds"] for item in records]
    tokens = [item["input_tokens"] / item["microstep_seconds"] for item in records if item["microstep_seconds"] > 0]
    return {
        "profile_records": len(records),
        "median_step_seconds": statistics.median(step_times) if step_times else 0.0,
        "p95_step_seconds": percentile(step_times, 0.95),
        "median_tokens_per_second": statistics.median(tokens) if tokens else 0.0,
        "mean_loader_wait_seconds": statistics.fmean([item.get("loader_wait_seconds", 0.0) for item in records]) if records else 0.0,
        "mean_h2d_seconds": statistics.fmean([item.get("h2d_seconds", 0.0) for item in records]) if records else 0.0,
        "mean_forward_seconds": statistics.fmean([item.get("forward_seconds", 0.0) for item in records]) if records else 0.0,
        "mean_backward_seconds": statistics.fmean([item.get("backward_seconds", 0.0) for item in records]) if records else 0.0,
        "mean_optimizer_seconds": statistics.fmean([item.get("optimizer_seconds", 0.0) for item in records]) if records else 0.0,
    }


def build_image(image: str) -> None:
    code = run(["docker", "build", "-f", "Dockerfile.train", "-t", image, "."], ROOT / "artifacts" / "last-docker-build.log")
    if code != 0:
        raise SystemExit(code)


def run_case(image: str, run_id: str, case: str, repeat: int, base_env: dict, sample_interval: float) -> dict:
    out_dir = ROOT / "artifacts" / "benchmarks" / run_id / case / f"repeat-{repeat:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data" / "perf-runs" / run_id / case / f"repeat-{repeat:02d}"
    prepare_data_dir(data_dir)
    env = os.environ.copy()
    env.update(base_env)
    env.update(CASES[case])
    env.update(
        {
            "DATA_DIR": "/app/data/perf-runs/" + f"{run_id}/{case}/repeat-{repeat:02d}",
            "TRAIN_COMMITTED_CORPUS_DIR": "/workspace/training/corpus/kimi-full-v1",
        }
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{ROOT / 'data'}:/app/data",
        "-v",
        f"{ROOT / 'training' / 'corpus'}:/workspace/training/corpus:ro",
        image,
        "train-scratch",
        "--preset",
        env.get("TRAIN_PRESET", "agent"),
    ]
    docker_env = []
    for key, value in sorted(env.items()):
        if key.startswith("TRAIN_") or key in {"DATA_DIR", "MODEL_NAME"}:
            docker_env.extend(["-e", f"{key}={value}"])
    command = command[:5] + docker_env + command[5:]
    started = time.time()
    with Telemetry(out_dir, sample_interval):
        code = run(command, out_dir / "trainer.log", env=os.environ.copy())
    elapsed = time.time() - started
    perf_path = data_dir / "runs" / "perf-steps.jsonl"
    summary = {
        "run_id": run_id,
        "case": case,
        "repeat": repeat,
        "returncode": code,
        "elapsed_seconds": elapsed,
        "env": {key: env[key] for key in sorted(env) if key.startswith("TRAIN_")},
        "data_dir": str(data_dir),
    }
    if perf_path.exists():
        shutil.copy2(perf_path, out_dir / "perf-steps.jsonl")
        summary.update(summarize_steps(perf_path))
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_summary(run_id: str, rows: list[dict]) -> None:
    out_dir = ROOT / "artifacts" / "benchmarks" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    aggregate = []
    for case, case_rows in grouped.items():
        medians = [row.get("median_step_seconds", 0.0) for row in case_rows if row.get("returncode") == 0]
        toks = [row.get("median_tokens_per_second", 0.0) for row in case_rows if row.get("returncode") == 0]
        aggregate.append(
            {
                "case": case,
                "runs": len(case_rows),
                "successful_runs": sum(1 for row in case_rows if row.get("returncode") == 0),
                "median_step_seconds": statistics.median(medians) if medians else 0.0,
                "median_tokens_per_second": statistics.median(toks) if toks else 0.0,
            }
        )
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--image", default="")
    parser.add_argument("--cases", default="real_legacy,real_mapped,synthetic_cpu,synthetic_gpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--max-optimizer-steps", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--warmup-microsteps", type=int, default=2)
    args = parser.parse_args()

    image = args.image or f"lkjai-train-perf:{args.run_id}"
    if not args.no_build:
        build_image(image)
    base_env = {
        "MODEL_NAME": "lkjai-scratch-60m",
        "TRAIN_PRESET": "agent",
        "TRAIN_MODEL_PRESET": "scratch-60m",
        "TRAIN_MAX_OPTIMIZER_STEPS": str(args.max_optimizer_steps),
        "TRAIN_MAX_STEPS": str(args.max_optimizer_steps),
        "TRAIN_GRADIENT_ACCUMULATION": str(args.gradient_accumulation),
        "TRAIN_VALIDATE_EVERY_OPTIMIZER_STEPS": "0",
        "TRAIN_SAVE_EVERY_OPTIMIZER_STEPS": "0",
        "TRAIN_VALIDATION_BATCHES": "1",
        "TRAIN_RESUME": "never",
        "TRAIN_PROFILE_STEPS": str(args.profile_steps),
        "TRAIN_BENCHMARK_WARMUP_MICROSTEPS": str(args.warmup_microsteps),
        "TRAIN_AMP": "auto",
        "TRAIN_ALLOW_TF32": "1",
        "TRAIN_MATMUL_PRECISION": "high",
    }
    rows = []
    for case in [item.strip() for item in args.cases.split(",") if item.strip()]:
        if case not in CASES:
            raise SystemExit(f"unknown case {case}")
        for repeat in range(1, args.repeats + 1):
            rows.append(run_case(image, args.run_id, case, repeat, base_env, args.sample_interval))
            write_summary(args.run_id, rows)
    print(json.dumps({"run_id": args.run_id, "rows": len(rows), "status": "pass"}))


if __name__ == "__main__":
    main()

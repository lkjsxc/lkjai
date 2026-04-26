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


def run(command: list[str], log_path: Path | None = None, env: dict | None = None) -> int:
    with subprocess.Popen(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
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
        target = dst / path.relative_to(src)
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
        totals.append(sum(values))
        idles.append(values[3] + values[4])
    return totals, idles


def cpu_percent(prev: tuple[list[int], list[int]], curr: tuple[list[int], list[int]]) -> list[float]:
    output = []
    for total_a, idle_a, total_b, idle_b in zip(prev[0], prev[1], curr[0], curr[1]):
        total_delta = max(1, total_b - total_a)
        output.append(100.0 * (1.0 - (idle_b - idle_a) / total_delta))
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
            import csv

            cpu_writer = csv.writer(cpu_file)
            cpu_writer.writerow(["timestamp", "mean_cpu", "max_cpu", "per_core"])
            while not self.stop.is_set():
                self._append_gpu(gpu_file)
                curr = read_proc_stat()
                values = cpu_percent(prev, curr)
                prev = curr
                cpu_writer.writerow([time.time(), statistics.fmean(values), max(values), " ".join(f"{v:.2f}" for v in values)])
                cpu_file.flush()
                mem_path.write_text(Path("/proc/meminfo").read_text(encoding="utf-8"), encoding="utf-8")
                pci = subprocess.run(["nvidia-smi", "pci", "-gCnt"], text=True, capture_output=True, check=False)
                pci_path.write_text(pci.stdout + pci.stderr, encoding="utf-8")
                self.stop.wait(self.interval)

    def _append_gpu(self, handle) -> None:
        result = subprocess.run(["nvidia-smi", f"--query-gpu={DEFAULT_GPU_QUERY}", "--format=csv"], text=True, capture_output=True, check=False)
        if result.returncode != 0:
            handle.write(result.stderr)
        elif handle.tell() == 0:
            handle.write(result.stdout)
        elif len(result.stdout.splitlines()) > 1:
            handle.write(result.stdout.splitlines()[-1] + "\n")
        handle.flush()


def summarize_steps(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    step_times = [item["microstep_seconds"] for item in records]
    tokens = [item["input_tokens"] / item["microstep_seconds"] for item in records if item["microstep_seconds"] > 0]
    mean = lambda key: statistics.fmean([item.get(key, 0.0) for item in records]) if records else 0.0
    return {
        "profile_records": len(records),
        "median_step_seconds": statistics.median(step_times) if step_times else 0.0,
        "p95_step_seconds": percentile(step_times, 0.95),
        "median_tokens_per_second": statistics.median(tokens) if tokens else 0.0,
        "mean_loader_wait_seconds": mean("loader_wait_seconds"),
        "mean_h2d_seconds": mean("h2d_seconds"),
        "mean_forward_seconds": mean("forward_seconds"),
        "mean_backward_seconds": mean("backward_seconds"),
        "mean_optimizer_seconds": mean("optimizer_seconds"),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] if low == high else ordered[low] * (high - rank) + ordered[high] * (rank - low)


def build_image(image: str) -> None:
    code = run(["docker", "build", "-f", "Dockerfile.train", "-t", image, "."], ROOT / "artifacts" / "last-docker-build.log")
    if code != 0:
        raise SystemExit(code)

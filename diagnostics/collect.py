#!/usr/bin/env python3
import argparse
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


COMMANDS = {
    "uname": ["uname", "-a"],
    "os_release": ["cat", "/etc/os-release"],
    "lscpu": ["lscpu"],
    "lsblk": ["lsblk", "-o", "NAME,TYPE,SIZE,MODEL,ROTA,TRAN,MOUNTPOINTS,FSTYPE"],
    "mount": ["mount"],
    "free": ["free", "-h"],
    "swaps": ["cat", "/proc/swaps"],
    "meminfo": ["cat", "/proc/meminfo"],
    "top": ["top", "-b", "-n", "1"],
    "top_threads": ["top", "-b", "-H", "-n", "1"],
    "dmesg": ["dmesg", "-T"],
    "journal_kernel_tail": ["journalctl", "-k", "--no-pager", "-n", "500"],
    "python3_version": ["python3", "--version"],
    "python3_pip_list": ["python3", "-m", "pip", "list"],
    "rustc_version": ["rustc", "--version"],
    "cargo_version": ["cargo", "--version"],
    "docker_version": ["docker", "--version"],
    "docker_compose_version": ["docker", "compose", "version"],
    "docker_info": ["docker", "info"],
    "nvidia_smi_L": ["nvidia-smi", "-L"],
    "nvidia_smi_q": ["nvidia-smi", "-q"],
    "nvidia_smi_clock": ["nvidia-smi", "-q", "-d", "CLOCK"],
    "nvidia_smi_power": ["nvidia-smi", "-q", "-d", "POWER"],
    "nvidia_smi_performance": ["nvidia-smi", "-q", "-d", "PERFORMANCE"],
    "nvidia_smi_temperature": ["nvidia-smi", "-q", "-d", "TEMPERATURE"],
    "nvidia_smi_query": [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,driver_version,vbios_version,pci.bus_id,compute_cap,"
        "pstate,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,"
        "clocks.current.graphics,clocks.current.sm,clocks.current.memory,clocks.max.graphics,"
        "clocks.max.sm,clocks.max.memory,power.draw,power.limit,temperature.gpu,"
        "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
        "--format=csv",
    ],
    "nvidia_smi_topo": ["nvidia-smi", "topo", "-m"],
    "nvidia_smi_pci_counters": ["nvidia-smi", "pci", "-gCnt"],
    "nvidia_smi_pci_errors": ["nvidia-smi", "pci", "-gErrCnt"],
    "nvidia_smi_nvlink": ["nvidia-smi", "nvlink", "-s"],
    "nvidia_smi_stats_help": ["nvidia-smi", "stats", "--help"],
    "nvcc_version": ["nvcc", "--version"],
    "nsys_path": ["which", "nsys"],
    "ncu_path": ["which", "ncu"],
    "perf_path": ["which", "perf"],
    "iostat_path": ["which", "iostat"],
    "pidstat_path": ["which", "pidstat"],
    "iotop_path": ["which", "iotop"],
    "fio_path": ["which", "fio"],
    "numactl_path": ["which", "numactl"],
    "lspci_gpu": ["bash", "-lc", "lspci -vv 2>/dev/null | grep -i -A40 -B5 nvidia"],
}


DOCKER_PYTORCH_COMMAND = [
    "docker",
    "run",
    "--rm",
    "--gpus",
    "all",
    "--entrypoint",
    "python",
    "lkjai-train:latest",
    "-c",
    (
        "import json, sys, torch, tokenizers; "
        "print(json.dumps({"
        "'python': sys.version, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda, "
        "'cudnn': torch.backends.cudnn.version(), 'cuda_available': torch.cuda.is_available(), "
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
        "'capability': torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None, "
        "'bf16': torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None, "
        "'tokenizers': tokenizers.__version__"
        "}, indent=2))"
    ),
]


def run(command: list[str], timeout: int) -> dict:
    started = time.time()
    if not shutil.which(command[0]):
        return {"status": "missing", "command": command, "elapsed_seconds": 0.0, "stdout": "", "stderr": ""}
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "status": "ok" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "command": command,
            "elapsed_seconds": time.time() - started,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "command": command,
            "elapsed_seconds": time.time() - started,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }


def write_record(out_dir: Path, name: str, record: dict) -> None:
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out_dir / f"{name}.txt").write_text(record.get("stdout", "") + record.get("stderr", ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    out_dir = ROOT / "artifacts" / "diagnostics" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": args.run_id,
        "cwd": str(ROOT),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "wsl": "microsoft-standard-WSL" in platform.release(),
        "environment": {key: os.environ.get(key, "") for key in ["PATH", "VIRTUAL_ENV", "CONDA_PREFIX"]},
        "commands": {},
    }
    for name, command in COMMANDS.items():
        record = run(command, args.timeout)
        summary["commands"][name] = {"status": record["status"], "returncode": record.get("returncode")}
        write_record(out_dir, name, record)
    docker_record = run(DOCKER_PYTORCH_COMMAND, args.timeout * 3)
    summary["commands"]["docker_pytorch"] = {
        "status": docker_record["status"],
        "returncode": docker_record.get("returncode"),
    }
    write_record(out_dir, "docker_pytorch", docker_record)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_human_summary(out_dir, summary)
    print(json.dumps({"run_id": args.run_id, "out_dir": str(out_dir), "status": "pass"}))


def first_line(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            return line.strip()
    return ""


def write_human_summary(out_dir: Path, summary: dict) -> None:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, data in sorted(summary["commands"].items()):
        rows.append(f"| `{name}` | {data.get('status')} |")
    text = f"""# Environment Summary

Diagnostics run: `{summary['run_id']}`

- Platform: `{summary['platform']}`
- WSL detected: `{summary['wsl']}`
- GPU: `{first_line(out_dir / 'nvidia_smi_L.txt') or 'not detected'}`
- OS: `{first_line(out_dir / 'os_release.txt') or 'not detected'}`

| Command | Status |
| --- | --- |
{chr(10).join(rows)}
"""
    (reports / "environment-summary.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

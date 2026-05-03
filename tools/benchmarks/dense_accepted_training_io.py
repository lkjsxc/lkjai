import json
import os
from pathlib import Path

from run_support import ROOT, run


CASE = "dense_accepted_training_1024"
RUN_PURPOSE = "accepted_training"
MODEL_NAME = "dense-accepted-training"
SOURCE_HOST = ROOT / "data" / "train" / "datasets" / "train.jsonl"
TOKENIZER_HOST = ROOT / "data" / "train" / "tokenizer" / "tokenizer.json"
CONFIG_HOST = ROOT / "configs" / "native" / "native_accepted_dense_bf16.json"
CONFIG_CONTAINER = "/workspace/configs/native/native_accepted_dense_bf16.json"
SEQ_LEN = 128
SEQUENCE_COUNT = 256
SEED = 20260503


def workspace_path(path: Path) -> str:
    return "/workspace/" + str(path.relative_to(ROOT))


def app_data_path(path: Path) -> str:
    return "/app/data/" + str(path.relative_to(ROOT / "data"))


def docker_command(image: str, entrypoint: str, env: dict, args: list[str]) -> list[str]:
    command = ["docker", "run", "--rm", "--gpus", "all"]
    for key, value in sorted(env.items()):
        command.extend(["-e", f"{key}={value}"])
    command.extend(
        [
            "-v",
            f"{ROOT / 'data'}:/app/data",
            "-v",
            f"{ROOT / 'configs'}:/workspace/configs:ro",
            "--entrypoint",
            entrypoint,
            image,
        ]
    )
    command.extend(args)
    return command


def run_json(command: list[str], log_path: Path) -> dict:
    code = run(command, log_path, os.environ.copy())
    payload = {"status": "fail", "returncode": code, "log": str(log_path)}
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            payload = json.loads(line)
            payload.setdefault("returncode", code)
            break
    if code != 0:
        payload["status"] = "fail"
    return payload


def write_payloads(out_dir: Path, payloads: dict) -> None:
    for name, payload in payloads.items():
        (out_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

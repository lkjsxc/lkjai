import json
import os
from pathlib import Path

from run_support import ROOT, run


def host_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def workspace_path(path: Path) -> str:
    path = path.resolve()
    try:
        return "/workspace/" + str(path.relative_to(ROOT))
    except ValueError as exc:
        raise SystemExit(f"path must be inside repo for verify compose: {path}") from exc


def app_data_path(path: Path) -> str:
    path = path.resolve()
    try:
        return "/app/data/" + str(path.relative_to(ROOT / "data"))
    except ValueError as exc:
        raise SystemExit(f"path must be inside repo data/: {path}") from exc


def config_container_path(path: Path) -> str:
    path = path.resolve()
    try:
        return "/workspace/configs/" + str(path.relative_to(ROOT / "configs"))
    except ValueError as exc:
        raise SystemExit(f"native config must be inside configs/: {path}") from exc


def build_image(image: str, cuda_arch_flags: str, out_dir: Path) -> None:
    command = [
        "docker",
        "build",
        "-f",
        "ops/docker/Dockerfile.native",
        "-t",
        image,
        "--build-arg",
        f"LKJAI_CUDA_ARCHS={cuda_arch_flags}",
        ".",
    ]
    code = run(command, out_dir / "docker-build.log", os.environ.copy())
    if code != 0:
        raise SystemExit(code)


def cargo_builder(builder_args: list[str], log_path: Path) -> None:
    command = [
        "docker",
        "compose",
        "--progress",
        "quiet",
        "--profile",
        "verify",
        "run",
        "--rm",
        "--entrypoint",
        "cargo",
        "verify",
        "run",
        "-p",
        "lkjai_packed_cache_builder",
        "--",
        *builder_args,
    ]
    code = run(command, log_path, os.environ.copy())
    if code != 0:
        raise SystemExit(code)


def build_and_validate_cache(args, out_dir: Path) -> dict:
    source = host_path(args.source)
    tokenizer = host_path(args.tokenizer)
    config = host_path(args.native_config)
    cache = host_path(args.cache)
    for path, label in (
        (source, "source"),
        (tokenizer, "tokenizer"),
        (config, "config"),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    base = [
        "--source",
        workspace_path(source),
        "--tokenizer",
        workspace_path(tokenizer),
        "--config",
        workspace_path(config),
    ]
    if not args.skip_cache_build:
        cargo_builder(
            [
                "build",
                *base,
                "--out",
                workspace_path(cache),
                "--split",
                "train",
                "--objective",
                "causal_lm_full",
                "--seq-len",
                str(args.seq_len),
                "--sequence-count",
                str(args.sequence_count),
                "--seed",
                str(args.seed),
                "--run-id",
                args.run_id,
            ],
            out_dir / "packed-cache-build.log",
        )
    cargo_builder(
        ["validate", "--cache", workspace_path(cache), *base],
        out_dir / "packed-cache-validate.log",
    )
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    (out_dir / "packed-cache-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


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

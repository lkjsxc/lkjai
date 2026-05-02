import json
import os
import time
from pathlib import Path

from benchmark_reports import (
    dense_promotion_errors,
    is_promotable_dense_summary,
    summarize_train_report,
)
from run_support import ROOT, Telemetry, build_image, load_train_report, run


CASE = "dense_40m_compat_4"
RUN_PURPOSE = "bounded_compatibility_start_check"
SOURCE_HOST = ROOT / "data" / "train" / "datasets" / "train.jsonl"
TOKENIZER_HOST = ROOT / "data" / "train" / "tokenizer" / "tokenizer.json"
CONFIG_HOST = ROOT / "configs" / "native" / "native_40m_bf16.json"
CONFIG_CONTAINER = "/workspace/configs/native/native_40m_bf16.json"


def workspace_path(path: Path) -> str:
    return "/workspace/" + str(path.relative_to(ROOT))


def app_data_path(path: Path) -> str:
    return "/app/data/" + str(path.relative_to(ROOT / "data"))


def cargo_builder(args: list[str], log_path: Path) -> None:
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
        *args,
    ]
    code = run(command, log_path, os.environ.copy())
    if code != 0:
        raise SystemExit(code)


def build_and_validate_cache(run_id: str, sequence_count: int, cache_dir: Path, out_dir: Path) -> None:
    base = [
        "--source",
        workspace_path(SOURCE_HOST),
        "--tokenizer",
        workspace_path(TOKENIZER_HOST),
        "--config",
        workspace_path(CONFIG_HOST),
    ]
    cargo_builder(
        [
            "build",
            *base,
            "--out",
            workspace_path(cache_dir),
            "--split",
            "train",
            "--objective",
            "causal_lm_full",
            "--seq-len",
            "1024",
            "--sequence-count",
            str(sequence_count),
            "--seed",
            "20260502",
            "--run-id",
            run_id,
        ],
        out_dir / "packed-cache-build.log",
    )
    cargo_builder(
        ["validate", "--cache", workspace_path(cache_dir), *base],
        out_dir / "packed-cache-validate.log",
    )


def docker_command(image: str, entrypoint: str, env: dict, args: list[str]) -> list[str]:
    command = ["docker", "run", "--rm", "--gpus", "all"]
    for key, value in sorted(env.items()):
        command.extend(["-e", f"{key}={value}"])
    command.extend(
        [
            "-v",
            f"{ROOT / 'data'}:/app/data",
            "-v",
            f"{ROOT / 'artifacts'}:/workspace/artifacts",
            "-v",
            f"{ROOT / 'configs'}:/workspace/configs:ro",
            "--entrypoint",
            entrypoint,
            image,
        ]
    )
    command.extend(args)
    return command


def run_train(image: str, steps: int, cache_dir: Path, repeat_dir: Path) -> int:
    env = {
        "DATA_DIR": workspace_path(repeat_dir),
        "MODEL_NAME": "dense-40m-compat",
        "TRAIN_BATCH_SIZE": "1",
        "TRAIN_GRADIENT_ACCUMULATION": "1",
        "TRAIN_LEARNING_RATE": "0.0003",
        "TRAIN_MAX_OPTIMIZER_STEPS": str(steps),
        "TRAIN_NATIVE_CONFIG": CONFIG_CONTAINER,
        "TRAIN_PACKED_CACHE_DIR": app_data_path(cache_dir),
        "TRAIN_RUN_PURPOSE": RUN_PURPOSE,
        "TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS": "4",
        "TRAIN_SEQUENCE_LEN": "1024",
    }
    args = [
        "--train", "--mode", "dense", "--run-purpose", RUN_PURPOSE,
        "--config", CONFIG_CONTAINER, "--packed-cache", app_data_path(cache_dir),
        "--out", workspace_path(repeat_dir), "--seq-len", "1024",
        "--batch-size", "1", "--grad-accum", "1", "--max-steps", str(steps),
        "--checkpoint-interval", "4", "--lr", "0.0003",
    ]
    return run(docker_command(image, "lkjai-native-train", env, args),
               repeat_dir / "train.log", os.environ.copy())


def run_logits_reference(image: str, report: dict, repeat_dir: Path) -> dict:
    export_path = report.get("export_path", "")
    checkpoint_path = report.get("checkpoint_path", "")
    if not export_path or not checkpoint_path:
        return {"status": "skipped", "reason": "missing export or checkpoint path"}
    log_path = repeat_dir / "logits-reference.log"
    code = run(
        docker_command(
            image,
            "lkjai-native-logits-check",
            {},
            ["--model-dir", export_path, "--tokens", "1,2,3",
             "--reference-checkpoint", checkpoint_path],
        ),
        log_path,
        os.environ.copy(),
    )
    if code != 0:
        return {"status": "fail", "returncode": code, "log": str(log_path)}
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            payload = json.loads(line)
            (repeat_dir / "logits-reference.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            return payload
    return {"status": "fail", "reason": "missing logits JSON", "log": str(log_path)}


def run_compat(args) -> dict:
    cache_dir = ROOT / "data" / "perf-runs" / args.run_id / "packed" / "train-causal_lm_full-seq1024"
    repeat_dir = ROOT / "artifacts" / "benchmarks" / args.run_id / CASE / "repeat-01"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    build_and_validate_cache(args.run_id, args.sequence_count, cache_dir, repeat_dir)
    build_image(args.image)
    started = time.monotonic()
    with Telemetry(repeat_dir, args.sample_interval):
        returncode = run_train(args.image, args.steps, cache_dir, repeat_dir)
    if returncode != 0:
        raise SystemExit(returncode)
    report = load_train_report(repeat_dir, repeat_dir / "train.log")
    if report.get("run_purpose") != RUN_PURPOSE:
        raise RuntimeError(f"unexpected run_purpose: {report.get('run_purpose')}")
    logits_reference = run_logits_reference(args.image, report, repeat_dir)
    summary = summarize_train_report(report)
    summary.update({
        "run_id": args.run_id, "case": CASE, "repeat": "repeat-01",
        "run_purpose": report.get("run_purpose", summary.get("run_purpose", "")),
        "promotion_status": "compatibility_only", "returncode": returncode,
        "wall_elapsed_seconds": time.monotonic() - started,
        "logits_reference": logits_reference,
        "promotion_errors": dense_promotion_errors(report),
        "promotable": is_promotable_dense_summary({**summary, "returncode": returncode}),
    })
    text = json.dumps(summary, indent=2) + "\n"
    (repeat_dir / "compatibility-summary.json").write_text(text, encoding="utf-8")
    (repeat_dir / "benchmark-summary.json").write_text(text, encoding="utf-8")
    return summary

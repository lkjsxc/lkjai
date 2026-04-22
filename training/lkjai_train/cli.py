import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .adapter import train_adapter
from .dataset import prepare_fixtures, validate_dataset
from .evals import evaluate_fixed_suite
from .manifest import export_manifest, train_adapter_manifest
from .paths import Paths


def main() -> None:
    parser = argparse.ArgumentParser(prog="lkjai-train")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/app/data"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-fixtures")
    validate = sub.add_parser("validate-dataset")
    validate.add_argument("--path", default="")
    train = sub.add_parser("train-adapter")
    train.add_argument("--preset", default=os.environ.get("TRAIN_PRESET", "quick"))
    fixed = sub.add_parser("fixed-eval")
    fixed.add_argument("--threshold", type=float, default=0.8)
    sub.add_parser("export-manifest")
    sub.add_parser("smoke")
    sub.add_parser("train")
    args = parser.parse_args()
    result = dispatch(args, Paths(args.data_dir))
    print(json.dumps({"command": args.command, "status": "pass", "result": str(result)}))


def dispatch(args, paths: Paths):
    if args.command == "prepare-fixtures":
        return prepare_fixtures(paths)
    if args.command == "validate-dataset":
        return validate_dataset(Path(args.path) if args.path else paths.fixtures)
    if args.command == "train-adapter":
        settings = train_settings(args.preset)
        train_adapter(paths, settings)
        return train_adapter_manifest(paths, settings)
    if args.command == "fixed-eval":
        return evaluate_fixed_suite(paths, args.threshold)
    if args.command == "export-manifest":
        return export_manifest(paths, train_settings(os.environ.get("TRAIN_PRESET", "quick")))
    if args.command == "smoke":
        return smoke(paths)
    if args.command == "train":
        return train_pipeline(paths)
    raise ValueError(args.command)


def smoke(paths: Paths):
    prepare_fixtures(paths)
    validate_dataset(paths.fixtures)
    settings = train_settings("quick")
    train_adapter(paths, settings)
    train_adapter_manifest(paths, settings)
    export_manifest(paths, settings)
    return evaluate_fixed_suite(paths, 0.0)


def train_pipeline(paths: Paths):
    preset = os.environ.get("TRAIN_PRESET", "quick")
    settings = train_settings(preset)
    prepare_fixtures(paths)
    validate_dataset(paths.fixtures)
    train_adapter(paths, settings)
    train_adapter_manifest(paths, settings)
    export_manifest(paths, settings)
    report = evaluate_fixed_suite(paths, settings.fixed_eval_threshold)
    data = json.loads(report.read_text(encoding="utf-8"))
    if settings.enforce_competency and data["pass_rate"] < settings.fixed_eval_threshold:
        raise RuntimeError("agent competency gate failed")
    return report


@dataclass
class TrainSettings:
    preset: str
    base_model: str
    sequence_len: int
    lora_rank: int
    lora_alpha: int
    learning_rate: str
    load_in_4bit: bool
    gradient_checkpointing: bool
    epochs: float
    batch_size: int
    gradient_accumulation: int
    max_steps: int
    eval_ratio: float
    lora_dropout: float
    fixed_eval_threshold: float
    enforce_competency: bool


def train_settings(preset: str) -> TrainSettings:
    if preset == "quick":
        return TrainSettings(
            preset,
            "Qwen/Qwen3-0.6B",
            512,
            4,
            8,
            "1e-4",
            True,
            True,
            1.0,
            1,
            1,
            0,
            0.5,
            0.05,
            0.8,
            False,
        )
    if preset in {"agent", "custom"}:
        return TrainSettings(
            preset,
            os.environ.get("TRAIN_BASE_MODEL", "Qwen/Qwen3-0.6B"),
            env_int("TRAIN_SEQUENCE_LEN", 2048),
            env_int("TRAIN_LORA_RANK", 16),
            env_int("TRAIN_LORA_ALPHA", 32),
            os.environ.get("TRAIN_LEARNING_RATE", "1e-4"),
            env_bool("TRAIN_LOAD_IN_4BIT", True),
            env_bool("TRAIN_GRADIENT_CHECKPOINTING", True),
            env_float("TRAIN_EPOCHS", 1.0),
            env_int("TRAIN_BATCH_SIZE", 1),
            env_int("TRAIN_GRADIENT_ACCUMULATION", 8),
            env_int("TRAIN_MAX_STEPS", 200),
            env_float("TRAIN_EVAL_RATIO", 0.2),
            env_float("TRAIN_LORA_DROPOUT", 0.05),
            env_float("TRAIN_FIXED_EVAL_THRESHOLD", 0.8),
            env_bool("TRAIN_ENFORCE_COMPETENCY", False),
        )
    raise ValueError(f"unknown TRAIN_PRESET={preset}")


def settings_json(settings: TrainSettings) -> str:
    return json.dumps(asdict(settings), indent=2)


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()

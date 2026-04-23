import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .dataset import prepare_corpus, prepare_fixtures, validate_dataset
from .evals import evaluate_fixed_suite
from .manifest import checkpoint_manifest, export_manifest
from .paths import Paths


def main() -> None:
    parser = argparse.ArgumentParser(prog="lkjai-train")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/app/data"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-fixtures")
    sub.add_parser("prepare-corpus")
    sub.add_parser("train-tokenizer")
    validate = sub.add_parser("validate-dataset")
    validate.add_argument("--path", default="")
    train = sub.add_parser("train-scratch")
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
    if args.command == "prepare-corpus":
        return prepare_corpus(paths, train_settings(env_preset()).corpus_size)
    if args.command == "train-tokenizer":
        return run_tokenizer(paths, train_settings(env_preset()))
    if args.command == "validate-dataset":
        return validate_dataset(Path(args.path) if args.path else default_dataset(paths))
    if args.command == "train-scratch":
        return run_training(paths, train_settings(args.preset))
    if args.command == "fixed-eval":
        return evaluate_fixed_suite(paths, args.threshold)
    if args.command == "export-manifest":
        return export_manifest(paths, train_settings(env_preset()))
    if args.command == "smoke":
        return smoke(paths)
    if args.command == "train":
        return train_pipeline(paths)
    raise ValueError(args.command)


def run_training(paths: Paths, settings):
    from .scratch_train import train_scratch

    train_scratch(paths, settings)
    return checkpoint_manifest(paths, settings)


def run_tokenizer(paths: Paths, settings):
    from .tokenizer import train_tokenizer

    return train_tokenizer(paths, settings)


def smoke(paths: Paths):
    settings = train_settings("quick")
    prepare_fixtures(paths)
    run_tokenizer(paths, settings)
    validate_dataset(paths.fixtures)
    run_training(paths, settings)
    export_manifest(paths, settings)
    return evaluate_fixed_suite(paths, 0.0)


def train_pipeline(paths: Paths):
    settings = train_settings(env_preset())
    dataset_path = prepare_corpus(paths, settings.corpus_size)
    run_tokenizer(paths, settings)
    validate_dataset(dataset_path)
    run_training(paths, settings)
    export_manifest(paths, settings)
    report = evaluate_fixed_suite(paths, settings.fixed_eval_threshold)
    data = json.loads(report.read_text(encoding="utf-8"))
    if settings.enforce_competency and data["pass_rate"] < settings.fixed_eval_threshold:
        raise RuntimeError("agent competency gate failed")
    return report


@dataclass
class TrainSettings:
    preset: str
    model_name: str
    model_preset: str
    vocab_size: int
    sequence_len: int
    layers: int
    hidden_size: int
    heads: int
    kv_heads: int
    ffn_size: int
    learning_rate: float
    gradient_checkpointing: bool
    batch_size: int
    gradient_accumulation: int
    max_steps: int
    fixed_eval_threshold: float
    enforce_competency: bool
    corpus_size: int
    seed: int


def train_settings(preset: str) -> TrainSettings:
    if preset == "quick":
        return settings("quick", "tiny-scratch", 512, 64, 2, 64, 4, 2, 128, 5, 20)
    if preset in {"agent", "custom"}:
        return settings(preset, env_str("TRAIN_MODEL_PRESET", "scratch-40m"), 8192, 1024, 8, 512, 8, 2, 1536, 500, 200)
    raise ValueError(f"unknown TRAIN_PRESET={preset}")


def settings(preset, model_preset, vocab, seq, layers, hidden, heads, kv, ffn, steps, rows):
    return TrainSettings(
        preset=preset,
        model_name=env_str("MODEL_NAME", "lkjai-scratch-40m"),
        model_preset=model_preset,
        vocab_size=env_int("TRAIN_VOCAB_SIZE", vocab),
        sequence_len=env_int("TRAIN_SEQUENCE_LEN", seq),
        layers=env_int("TRAIN_LAYERS", layers),
        hidden_size=env_int("TRAIN_HIDDEN_SIZE", hidden),
        heads=env_int("TRAIN_HEADS", heads),
        kv_heads=env_int("TRAIN_KV_HEADS", kv),
        ffn_size=env_int("TRAIN_FFN_SIZE", ffn),
        learning_rate=env_float("TRAIN_LEARNING_RATE", 3e-4),
        gradient_checkpointing=env_bool("TRAIN_GRADIENT_CHECKPOINTING", True),
        batch_size=env_int("TRAIN_BATCH_SIZE", 1),
        gradient_accumulation=env_int("TRAIN_GRADIENT_ACCUMULATION", 8),
        max_steps=env_int("TRAIN_MAX_STEPS", steps),
        fixed_eval_threshold=env_float("TRAIN_FIXED_EVAL_THRESHOLD", 0.8),
        enforce_competency=env_bool("TRAIN_ENFORCE_COMPETENCY", False),
        corpus_size=env_int("TRAIN_CORPUS_SIZE", rows),
        seed=env_int("TRAIN_SEED", 42),
    )


def default_dataset(paths: Paths) -> Path:
    return paths.corpus if paths.corpus.exists() else paths.fixtures


def env_preset() -> str:
    return os.environ.get("TRAIN_PRESET", "quick")


def env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()

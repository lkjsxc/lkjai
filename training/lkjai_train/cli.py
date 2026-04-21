import argparse
import json
import os
import sys
from dataclasses import dataclass

from .corpus import prepare_corpus
from .export import export_model
from .fixed_eval import evaluate_fixed_suite
from .packer import pack_tokens
from .paths import Paths
from .tokenizer import train_tokenizer
from .trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="lkjai-train")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/app/data"))
    sub = parser.add_subparsers(dest="command", required=True)
    corpus = sub.add_parser("prepare-corpus")
    corpus.add_argument("--token-budget", type=int, default=3_000_000_000)
    corpus.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    corpus.add_argument("--tiny", action="store_true")
    tokenizer = sub.add_parser("train-tokenizer")
    tokenizer.add_argument("--vocab-size", type=int, default=32_000)
    sub.add_parser("pack-tokens")
    train = sub.add_parser("train-model")
    train.add_argument("--config", default="")
    train.add_argument("--context", type=int, default=0)
    train.add_argument("--tiny", action="store_true")
    train.add_argument("--steps", type=int, default=8)
    train.add_argument("--max-duration-secs", type=int, default=0)
    export = sub.add_parser("export-model")
    export.add_argument("--max-artifact-mib", type=int, default=512)
    fixed = sub.add_parser("fixed-eval")
    fixed.add_argument("--threshold", type=float, default=0.8)
    sub.add_parser("smoke")
    sub.add_parser("train")
    args = parser.parse_args()
    paths = Paths(args.data_dir)
    result = dispatch(args, paths)
    print(json.dumps({"command": args.command, "status": "pass", "result": str(result)}))
    if args.command == "prepare-corpus":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def dispatch(args, paths):
    if args.command == "prepare-corpus":
        return prepare_corpus(paths, args.token_budget, args.dataset, args.tiny)
    if args.command == "train-tokenizer":
        return train_tokenizer(paths, args.vocab_size)
    if args.command == "pack-tokens":
        return pack_tokens(paths)
    if args.command == "train-model":
        return train_model(
            paths,
            args.tiny,
            args.steps,
            args.config,
            args.context,
            args.max_duration_secs,
        )
    if args.command == "export-model":
        return export_model(paths, args.max_artifact_mib)
    if args.command == "fixed-eval":
        return evaluate_fixed_suite(paths, args.threshold)
    if args.command == "smoke":
        prepare_corpus(paths, 200, "fixture", True)
        train_tokenizer(paths, 259)
        pack_tokens(paths)
        train_model(paths, True, 2)
        result = export_model(paths, 512)
        evaluate_fixed_suite(paths, 0.0)
        return result
    if args.command == "train":
        return train_pipeline(paths)
    raise ValueError(args.command)


def train_pipeline(paths):
    preset = os.environ.get("TRAIN_PRESET", "quick")
    settings = train_settings(preset)
    log(
        f"train preset={preset} tiny={settings.tiny} budget={settings.token_budget} "
        f"steps={settings.steps} duration={settings.max_duration_secs}"
    )
    prepare_corpus(paths, settings.token_budget, settings.dataset, settings.tiny)
    log("corpus prepared")
    train_tokenizer(paths, settings.vocab_size)
    log("tokenizer trained")
    pack_tokens(paths)
    log("tokens packed")
    train_model(
        paths,
        settings.tiny,
        settings.steps,
        settings.config,
        settings.context,
        settings.max_duration_secs,
    )
    log("model trained")
    out = export_model(paths, env_int("TRAIN_MAX_ARTIFACT_MIB", 512))
    report_path = evaluate_fixed_suite(paths, settings.fixed_eval_threshold)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pass_rate = float(report.get("pass_rate", 0.0))
    log(f"fixed-eval pass_rate={pass_rate:.4f} threshold={settings.fixed_eval_threshold:.4f}")
    if settings.enforce_competency and pass_rate < settings.fixed_eval_threshold:
        raise RuntimeError(
            f"competency gate failed: pass_rate={pass_rate:.4f} "
            f"< threshold={settings.fixed_eval_threshold:.4f}"
        )
    return out


@dataclass
class TrainSettings:
    tiny: bool
    token_budget: int
    dataset: str
    vocab_size: int
    steps: int
    context: int
    config: str
    max_duration_secs: int
    fixed_eval_threshold: float
    enforce_competency: bool


def train_settings(preset: str) -> TrainSettings:
    if preset == "quick":
        return TrainSettings(True, 200, "fixture", 259, 2, 32, "", 0, 0.8, False)
    if preset == "longrun":
        return TrainSettings(
            False,
            env_int("TRAIN_TOKEN_BUDGET", 3_000_000_000),
            os.environ.get("TRAIN_DATASET", "HuggingFaceFW/fineweb-edu"),
            env_int("TRAIN_VOCAB_SIZE", 32_000),
            env_int("TRAIN_STEPS", 512),
            env_int("TRAIN_CONTEXT", 0),
            os.environ.get("TRAIN_CONFIG", "/workspace/configs/lkj-150m.toml"),
            env_int("TRAIN_MAX_DURATION_SECS", 21_600),
            env_float("TRAIN_FIXED_EVAL_THRESHOLD", 0.8),
            env_bool("TRAIN_ENFORCE_COMPETENCY", True),
        )
    if preset == "full":
        return TrainSettings(
            False,
            env_int("TRAIN_TOKEN_BUDGET", 3_000_000_000),
            os.environ.get("TRAIN_DATASET", "HuggingFaceFW/fineweb-edu"),
            env_int("TRAIN_VOCAB_SIZE", 32_000),
            env_int("TRAIN_STEPS", 512),
            env_int("TRAIN_CONTEXT", 0),
            os.environ.get("TRAIN_CONFIG", "/workspace/configs/lkj-150m.toml"),
            env_int("TRAIN_MAX_DURATION_SECS", 0),
            env_float("TRAIN_FIXED_EVAL_THRESHOLD", 0.8),
            env_bool("TRAIN_ENFORCE_COMPETENCY", False),
        )
    if preset == "custom":
        return TrainSettings(
            env_bool("TRAIN_TINY", False),
            env_int("TRAIN_TOKEN_BUDGET", 3_000_000_000),
            os.environ.get("TRAIN_DATASET", "HuggingFaceFW/fineweb-edu"),
            env_int("TRAIN_VOCAB_SIZE", 32_000),
            env_int("TRAIN_STEPS", 8),
            env_int("TRAIN_CONTEXT", 0),
            os.environ.get("TRAIN_CONFIG", "/workspace/configs/lkj-150m.toml"),
            env_int("TRAIN_MAX_DURATION_SECS", 0),
            env_float("TRAIN_FIXED_EVAL_THRESHOLD", 0.8),
            env_bool("TRAIN_ENFORCE_COMPETENCY", False),
        )
    raise ValueError(f"unknown TRAIN_PRESET={preset}")


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def log(message: str) -> None:
    print(json.dumps({"event": message}), flush=True)


if __name__ == "__main__":
    main()

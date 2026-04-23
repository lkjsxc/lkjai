import argparse
import json
import os
from pathlib import Path

from .behavioral import evaluate_behavior
from .dataset import prepare_corpus, prepare_fixtures, validate_dataset
from .evals import evaluate_fixed_suite
from .manifest import checkpoint_manifest, export_manifest
from .paths import Paths
from .preference import mark_dpo_rejected, prepare_preferences, train_dpo
from .settings import train_settings


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
    behavioral = sub.add_parser("behavioral-eval")
    behavioral.add_argument("--threshold", type=float, default=0.8)
    sub.add_parser("prepare-preferences")
    dpo = sub.add_parser("train-dpo")
    dpo.add_argument("--preset", default=os.environ.get("TRAIN_PRESET", "quick"))
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
    if args.command == "behavioral-eval":
        return evaluate_behavior(paths, train_settings(env_preset()), args.threshold)
    if args.command == "prepare-preferences":
        return prepare_preferences(paths)
    if args.command == "train-dpo":
        return train_dpo(paths, train_settings(args.preset))
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
    prepare_fixtures(paths)
    dataset_path = prepare_corpus(paths, settings.corpus_size)
    run_tokenizer(paths, settings)
    validate_dataset(dataset_path)
    run_training(paths, settings)
    export_manifest(paths, settings)
    report = evaluate_fixed_suite(paths, settings.fixed_eval_threshold)
    behavioral = evaluate_behavior(paths, settings, settings.fixed_eval_threshold)
    before = pass_rate(behavioral)
    prepare_preferences(paths)
    train_dpo(paths, settings)
    export_manifest(paths, settings)
    behavioral = evaluate_behavior(paths, settings, settings.fixed_eval_threshold)
    if pass_rate(behavioral) < before:
        mark_dpo_rejected(paths, "post-DPO behavioral pass rate regressed")
        export_manifest(paths, settings)
        behavioral = evaluate_behavior(paths, settings, settings.fixed_eval_threshold)
    data = json.loads(report.read_text(encoding="utf-8"))
    behavior = json.loads(behavioral.read_text(encoding="utf-8"))
    if settings.enforce_competency and behavior["pass_rate"] < settings.fixed_eval_threshold:
        raise RuntimeError("agent competency gate failed")
    if data["pass_rate"] < settings.fixed_eval_threshold:
        raise RuntimeError("fixed artifact gate failed")
    return behavioral


def default_dataset(paths: Paths) -> Path:
    return paths.corpus if paths.corpus.exists() else paths.fixtures


def env_preset() -> str:
    return os.environ.get("TRAIN_PRESET", "quick")


def pass_rate(path: Path) -> float:
    return float(json.loads(path.read_text(encoding="utf-8")).get("pass_rate", 0.0))


if __name__ == "__main__":
    main()

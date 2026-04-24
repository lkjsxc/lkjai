import argparse
import json
import os
from pathlib import Path

from .corpus_source import SOURCE_DIR, validate_sources
from .dataset import prepare_corpus, prepare_fixtures, validate_dataset
from .manifest import checkpoint_manifest, export_manifest
from .paths import Paths
from .settings import train_settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="lkjai-train")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/app/data"))
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["validate-sources", "validate-public-sources", "prepare-fixtures", "prepare-corpus", "prepare-public-corpus", "train-tokenizer", "prepare-preferences", "export-manifest", "smoke", "train"]:
        sub.add_parser(command)
    validate = sub.add_parser("validate-dataset")
    validate.add_argument("--path", default="")
    train = sub.add_parser("train-scratch")
    train.add_argument("--preset", default=os.environ.get("TRAIN_PRESET", "quick"))
    for name in ["fixed-eval", "behavioral-eval"]:
        parser_eval = sub.add_parser(name)
        parser_eval.add_argument("--threshold", type=float, default=0.0)
    dpo = sub.add_parser("train-dpo")
    dpo.add_argument("--preset", default=os.environ.get("TRAIN_PRESET", "quick"))
    args = parser.parse_args()
    result = dispatch(args, Paths(args.data_dir))
    print(json.dumps({"command": args.command, "status": "pass", "result": str(result)}))


def dispatch(args, paths: Paths):
    if args.command == "prepare-fixtures":
        return prepare_fixtures(paths)
    if args.command == "validate-sources":
        validate_sources()
        return SOURCE_DIR
    if args.command == "validate-public-sources":
        from .public_import import validate_public_sources

        return validate_public_sources(paths)
    if args.command == "prepare-corpus":
        return prepare_corpus(paths, train_settings(env_preset()).corpus_size)
    if args.command == "prepare-public-corpus":
        from .public_import import prepare_public_corpus

        return prepare_public_corpus(paths)
    if args.command == "train-tokenizer":
        return run_tokenizer(paths, train_settings(env_preset()))
    if args.command == "validate-dataset":
        return validate_dataset(Path(args.path) if args.path else default_dataset(paths))
    if args.command == "train-scratch":
        return run_training(paths, train_settings(args.preset))
    if args.command == "fixed-eval":
        from .evals import evaluate_fixed_suite

        return evaluate_fixed_suite(paths, args.threshold)
    if args.command == "behavioral-eval":
        from .behavioral import evaluate_behavior

        settings = train_settings(env_preset())
        return evaluate_behavior(paths, settings, args.threshold or settings.behavioral_threshold)
    if args.command == "prepare-preferences":
        from .preference import prepare_preferences

        return prepare_preferences(paths)
    if args.command == "train-dpo":
        from .preference import train_dpo

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
    from .evals import evaluate_fixed_suite

    settings = train_settings("quick")
    prepare_fixtures(paths)
    run_tokenizer(paths, settings)
    validate_dataset(paths.fixtures)
    run_training(paths, settings)
    export_manifest(paths, settings)
    return evaluate_fixed_suite(paths, 0.0)


def train_pipeline(paths: Paths):
    from .behavioral import evaluate_behavior
    from .evals import evaluate_fixed_suite

    settings = train_settings(env_preset())
    validate_sources()
    prepare_fixtures(paths)
    dataset_path = prepare_corpus(paths, settings.corpus_size)
    run_tokenizer(paths, settings)
    for path in [dataset_path, paths.train_dataset, paths.val_dataset, paths.holdout_dataset]:
        validate_dataset(path)
    run_training(paths, settings)
    export_manifest(paths, settings)
    fixed = evaluate_fixed_suite(paths, settings.fixed_eval_threshold)
    behavioral = evaluate_behavior(paths, settings, settings.behavioral_threshold)
    if settings.enforce_competency and not competency_passes(behavioral, settings.behavioral_threshold):
        raise RuntimeError("agent competency gate failed")
    if not fixed_artifact_passes(fixed, settings.fixed_eval_threshold):
        raise RuntimeError(f"fixed artifact gate failed: {failed_case_ids(fixed)}")
    return behavioral


def competency_passes(path: Path, threshold: float) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("json_validity", 0.0) >= 0.95 and data.get("pass_rate", 0.0) >= threshold


def default_dataset(paths: Paths) -> Path:
    return paths.corpus if paths.corpus.exists() else paths.fixtures


def env_preset() -> str:
    return os.environ.get("TRAIN_PRESET", "quick")


def pass_rate(path: Path) -> float:
    return float(json.loads(path.read_text(encoding="utf-8")).get("pass_rate", 0.0))


def fixed_artifact_passes(path: Path, threshold: float) -> bool:
    return pass_rate(path) >= threshold


def failed_case_ids(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    failed = [item["id"] for item in data.get("cases", []) if not item.get("passed")]
    return ", ".join(failed) if failed else "pass_rate below threshold"


if __name__ == "__main__":
    main()

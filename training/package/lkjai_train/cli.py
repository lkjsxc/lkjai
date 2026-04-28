import argparse
import json
import os
from pathlib import Path

from .corpus_source import SOURCE_DIR, validate_sources
from .dataset import prepare_corpus, prepare_fixtures, validate_dataset
from .kimi_dataset import validate_kimi_corpus
from .manifest import checkpoint_manifest, export_manifest
from .paths import Paths
from .pipeline import failed_case_ids, fixed_artifact_passes, train_pipeline, train_sft
from .settings import train_settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="lkjai-train")
    parser.add_argument("--data-dir", default=os.environ.get("TRAIN_DATA_DIR", os.environ.get("DATA_DIR", "/app/data")))
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["validate-sources", "validate-public-sources", "validate-public-pretrain", "download-public-pretrain", "prepare-fixtures", "prepare-corpus", "prepare-public-corpus", "prepare-public-pretrain", "train-tokenizer", "prepare-preferences", "export-manifest", "smoke", "train"]:
        sub.add_parser(command)
    validate = sub.add_parser("validate-dataset")
    validate.add_argument("--path", default="")
    train = sub.add_parser("train-scratch")
    train.add_argument("--preset", default=os.environ.get("TRAIN_PRESET", "quick"))
    sft = sub.add_parser("train-sft")
    sft.add_argument("--preset", default=os.environ.get("TRAIN_PRESET", "quick"))
    for name in ["fixed-eval", "behavioral-eval"]:
        parser_eval = sub.add_parser(name)
        parser_eval.add_argument("--threshold", type=float, default=0.0)
        if name == "behavioral-eval":
            parser_eval.add_argument("--checkpoint", choices=["export", "best", "final"], default=os.environ.get("TRAIN_BEHAVIORAL_CHECKPOINT", "export"))
    sanity = sub.add_parser("generation-sanity")
    sanity.add_argument("--checkpoint", choices=["export", "best", "final"], default=os.environ.get("TRAIN_BEHAVIORAL_CHECKPOINT", "export"))
    dpo = sub.add_parser("train-dpo")
    dpo.add_argument("--preset", default=os.environ.get("TRAIN_PRESET", "quick"))
    simpo = sub.add_parser("train-simpo")
    simpo.add_argument("--preset", default=os.environ.get("TRAIN_PRESET", "quick"))
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
    if args.command == "validate-public-pretrain":
        from .public_pretrain import validate_public_pretrain_sources

        return validate_public_pretrain_sources(paths)
    if args.command == "download-public-pretrain":
        from .public_pretrain_download import download_public_pretrain

        return download_public_pretrain(paths)
    if args.command == "prepare-corpus":
        corpus = prepare_corpus(paths, train_settings(env_preset()).corpus_size)
        from .public_pretrain import validate_public_pretrain_sources

        validate_public_pretrain_sources(paths)
        return corpus
    if args.command == "prepare-public-corpus":
        from .public_import import prepare_public_corpus

        return prepare_public_corpus(paths)
    if args.command == "prepare-public-pretrain":
        from .public_pretrain import prepare_public_pretrain

        return prepare_public_pretrain(paths)
    if args.command == "train-tokenizer":
        return run_tokenizer(paths, train_settings(env_preset()))
    if args.command == "validate-dataset":
        if args.path:
            return validate_dataset(Path(args.path))
        if paths.kimi_corpus.exists() and any(paths.kimi_corpus.rglob("*.jsonl")):
            return validate_kimi_corpus(paths)
        return validate_dataset(default_dataset(paths))
    if args.command == "train-scratch":
        return run_training(paths, train_settings(args.preset))
    if args.command == "train-sft":
        return train_sft(paths, train_settings(args.preset))
    if args.command == "fixed-eval":
        from .evals import evaluate_fixed_suite

        return evaluate_fixed_suite(paths, args.threshold)
    if args.command == "behavioral-eval":
        from .behavioral import evaluate_behavior

        settings = train_settings(env_preset())
        return evaluate_behavior(paths, settings, args.threshold or settings.behavioral_threshold, args.checkpoint)
    if args.command == "generation-sanity":
        from .evals import evaluate_generation_sanity

        return evaluate_generation_sanity(paths, train_settings(env_preset()), args.checkpoint)
    if args.command == "prepare-preferences":
        from .preference import prepare_preferences

        return prepare_preferences(paths)
    if args.command == "train-dpo":
        from .preference import train_dpo

        return train_dpo(paths, train_settings(args.preset))
    if args.command == "train-simpo":
        from .preference import train_simpo

        return train_simpo(paths, train_settings(args.preset))
    if args.command == "export-manifest":
        return export_manifest(paths, train_settings(env_preset()))
    if args.command == "smoke":
        return smoke(paths)
    if args.command == "train":
        return train_pipeline(paths, train_settings(env_preset()))
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


def default_dataset(paths: Paths) -> Path:
    if paths.committed_kimi_corpus.exists() and any(paths.committed_kimi_corpus.rglob("*.jsonl")):
        return paths.committed_kimi_corpus
    if paths.kimi_corpus.exists() and any(paths.kimi_corpus.rglob("*.jsonl")):
        return paths.kimi_corpus
    return paths.corpus if paths.corpus.exists() else paths.fixtures


def env_preset() -> str:
    return os.environ.get("TRAIN_PRESET", "quick")


if __name__ == "__main__":
    main()

import argparse
import json
import os

from .corpus import prepare_corpus
from .export import export_model
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
    train.add_argument("--tiny", action="store_true")
    train.add_argument("--steps", type=int, default=8)
    export = sub.add_parser("export-model")
    export.add_argument("--max-artifact-mib", type=int, default=512)
    sub.add_parser("smoke")
    args = parser.parse_args()
    paths = Paths(args.data_dir)
    result = dispatch(args, paths)
    print(json.dumps({"command": args.command, "status": "pass", "result": str(result)}))


def dispatch(args, paths):
    if args.command == "prepare-corpus":
        return prepare_corpus(paths, args.token_budget, args.dataset, args.tiny)
    if args.command == "train-tokenizer":
        return train_tokenizer(paths, args.vocab_size)
    if args.command == "pack-tokens":
        return pack_tokens(paths)
    if args.command == "train-model":
        return train_model(paths, args.tiny, args.steps)
    if args.command == "export-model":
        return export_model(paths, args.max_artifact_mib)
    if args.command == "smoke":
        prepare_corpus(paths, 200, "fixture", True)
        train_tokenizer(paths, 259)
        pack_tokens(paths)
        train_model(paths, True, 2)
        return export_model(paths, 512)
    raise ValueError(args.command)


if __name__ == "__main__":
    main()

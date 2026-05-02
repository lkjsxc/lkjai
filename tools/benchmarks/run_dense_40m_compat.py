#!/usr/bin/env python3
import argparse
import json

from dense_40m_compat_support import (
    SOURCE_HOST,
    TOKENIZER_HOST,
    run_compat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--sequence-count", type=int, default=8)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--image", default="lkjai-native-bench")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps != 4:
        raise SystemExit("dense 40M compatibility runner is bounded to --steps 4")
    if args.sequence_count < 4:
        raise SystemExit("--sequence-count must provide at least four 1024-token windows")
    if not SOURCE_HOST.is_file():
        raise SystemExit(f"missing source JSONL: {SOURCE_HOST}")
    if not TOKENIZER_HOST.is_file():
        raise SystemExit(f"missing tokenizer: {TOKENIZER_HOST}")
    print(json.dumps(run_compat(args), indent=2))


if __name__ == "__main__":
    main()

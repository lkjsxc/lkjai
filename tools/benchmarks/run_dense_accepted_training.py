#!/usr/bin/env python3
import argparse
import json

from dense_accepted_training_support import SEQUENCE_COUNT, run_accepted
from run_support import build_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image", default="lkjai-native-bench")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--sequence-count", type=int, default=SEQUENCE_COUNT)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--loss-sample-interval", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--no-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sequence_count < 32:
        raise SystemExit("--sequence-count must be at least 32")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if not args.no_build:
        build_image(args.image)
    summary = run_accepted(args)
    print(json.dumps({"runner_status": "pass", **summary}))


if __name__ == "__main__":
    main()

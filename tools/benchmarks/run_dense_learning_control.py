#!/usr/bin/env python3
import argparse
import json
import time

from dense_learning_control_support import run_control
from run_support import build_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id", default=time.strftime("dense-learning-control-%Y%m%d-%H%M%S")
    )
    parser.add_argument("--image", default="")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--loss-sample-interval", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--no-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.image = args.image or f"lkjai-dense-learning-control:{args.run_id}"
    if not args.no_build:
        build_image(args.image)
    summary = run_control(args)
    print(json.dumps({"runner_status": "pass", **summary}))


if __name__ == "__main__":
    main()

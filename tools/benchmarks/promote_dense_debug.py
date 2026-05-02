#!/usr/bin/env python3
import argparse
import json
import time

from dense_debug_runner import run_promotion
from run_support import build_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=time.strftime("dense-debug-%Y%m%d-%H%M%S"))
    parser.add_argument("--image", default="")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--resume-steps", type=int, default=1)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    args.image = args.image or f"lkjai-dense-debug-promote:{args.run_id}"
    if not args.no_build:
        build_image(args.image)
    summary = run_promotion(args, args.steps)
    print(json.dumps({"runner_status": "pass", **summary}))


if __name__ == "__main__":
    main()

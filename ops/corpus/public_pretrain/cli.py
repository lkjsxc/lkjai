import argparse

from .download import download
from .prepare import prepare
from .validate import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "prepare", "validate"])
    args = parser.parse_args()
    if args.command == "download":
        download(args)
    elif args.command == "prepare":
        prepare(args)
    else:
        validate(args)

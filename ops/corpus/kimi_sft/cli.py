import argparse
import json

from .api import call_kimi, parse_kimi_rows
from .common import load_api_key, read_config
from .constants import DEFAULT_CONFIG, DEFAULT_PROMOTED, DEFAULT_QUARANTINE
from .constants import SCHEMA, TOOLS
from .generate import generate
from .promote import manifest_for, promote, report
from .validate import action_tools_for_messages, validate, validate_row

__all__ = [
    "SCHEMA",
    "TOOLS",
    "action_tools_for_messages",
    "call_kimi",
    "generate",
    "load_api_key",
    "manifest_for",
    "parse_kimi_rows",
    "promote",
    "read_config",
    "report",
    "validate",
    "validate_row",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "validate", "promote", "report"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--promoted", default=DEFAULT_PROMOTED)
    args = parser.parse_args()
    if args.command == "generate":
        result = generate(args)
    elif args.command == "promote":
        result = promote(args)
    elif args.command == "report":
        result = report(args)
    else:
        result = validate(args.quarantine, write_report=True)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

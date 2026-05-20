import argparse
import json
import shutil
from pathlib import Path

TOOLS = {
    "agent.finish",
    "agent.think",
    "agent.request_confirmation",
    "resource.search",
    "resource.get",
    "resource.history",
    "resource.create",
    "resource.update_resource",
    "resource.delete",
}


def rows(root):
    for path in sorted(Path(root).glob("*/*.jsonl")):
        with path.open(encoding="utf-8") as file:
            for number, line in enumerate(file, 1):
                if line.strip():
                    yield path, number, json.loads(line)


def validate_row(row, split, seen):
    meta = row.get("meta", {})
    messages = row.get("messages", [])
    errors = []
    if meta.get("schema") != "lkjai-agent-jsonl":
        errors.append("bad schema")
    if meta.get("split") != split:
        errors.append("split mismatch")
    row_id = meta.get("id")
    if not row_id or row_id in seen:
        errors.append("duplicate or missing id")
    seen.add(row_id)
    targets = [m for m in messages if m.get("role") == "assistant"]
    action_tools = []
    for target in targets:
        content = target.get("content", "")
        if content.count("<action>") != 1 or content.count("</action>") != 1:
            errors.append("assistant target must contain one action")
            continue
        tool = content.split("<tool>", 1)[-1].split("</tool>", 1)[0]
        action_tools.append(tool)
        if tool not in TOOLS:
            errors.append("tool not allowed")
    if not action_tools:
        errors.append("missing assistant action target")
    if meta.get("confirmation_required") and "agent.request_confirmation" not in action_tools:
        errors.append("confirmation row must request confirmation")
    return errors


def validate(root):
    seen = set()
    errors = []
    count = 0
    for path, number, row in rows(root):
        split = path.parent.name
        row_errors = validate_row(row, split, seen)
        count += 1
        for error in row_errors:
            errors.append(f"{path}:{number}: {error}")
    return {"status": "pass" if not errors else "fail", "rows": count, "errors": errors}


def generate(args):
    out = Path(args.quarantine) / "train"
    out.mkdir(parents=True, exist_ok=True)
    shard = out / "shard-000001.jsonl"
    if not shard.exists():
        shard.write_text("", encoding="utf-8")
    return {"status": "pass", "quarantine": args.quarantine, "note": "ready"}


def promote(args):
    report = validate(args.quarantine)
    if report["status"] != "pass":
        return report
    src = Path(args.quarantine)
    dst = Path(args.promoted)
    if dst.exists():
        shutil.rmtree(dst)
    for shard in src.glob("*/*.jsonl"):
        target = dst / shard.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shard, target)
    return {"status": "pass", "promoted": str(dst), "rows": report["rows"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "validate", "promote", "report"])
    parser.add_argument("--quarantine", default="data/corpus/quarantine/kimi-sft-60m")
    parser.add_argument("--promoted", default="data/corpus/generated/kimi-sft-60m")
    args = parser.parse_args()
    if args.command == "generate":
      result = generate(args)
    elif args.command == "promote":
      result = promote(args)
    else:
      root = args.promoted if args.command == "report" else args.quarantine
      result = validate(root)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

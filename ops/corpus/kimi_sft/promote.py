import shutil
from pathlib import Path

from .common import file_sha256, source_digest, write_json
from .constants import CORPUS, SCHEMA, SPLITS
from .validate import validate


def manifest_for(root, validation_report, status):
    shards = []
    for path in sorted(Path(root).glob("*/*.jsonl")):
        shards.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema": SCHEMA,
        "corpus": CORPUS,
        "status": status,
        "rows": validation_report.get("rows", 0),
        "path": str(root),
        "split_rows": validation_report.get("split_rows", {split: 0 for split in SPLITS}),
        "token_estimate": validation_report.get("token_estimate", 0),
        "provenance": "kimi-generated",
        "source_digest": source_digest(root),
        "validation_report": "validation-report.json",
        "shards": shards,
    }


def promote(args):
    report = validate(args.quarantine, write_report=True)
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
    validation = validate(dst, write_report=True)
    write_json(dst / "manifest.json", manifest_for(dst, validation, "generated"))
    return {"status": "pass", "promoted": str(dst), "rows": validation["rows"]}


def report(args):
    root = Path(args.promoted)
    validation = validate(root, write_report=False)
    manifest_path = root / "manifest.json"
    result = {
        "status": validation["status"],
        "promoted": str(root),
        "rows": validation["rows"],
        "split_rows": validation["split_rows"],
        "token_estimate": validation["token_estimate"],
        "manifest": str(manifest_path) if manifest_path.is_file() else "",
    }
    if validation["errors"]:
        result["errors"] = validation["errors"]
    return result

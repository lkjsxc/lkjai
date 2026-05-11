import json
from pathlib import Path

from .common import DEFAULT_OUT, TOKEN_RE, env_path, write_json


def iter_jsonl(root):
    for split in ("train", "val", "holdout"):
        for path in sorted((Path(root) / split).glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    if line.strip():
                        yield split, path, line_no, json.loads(line)


def check_no_forbidden_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"prompt", "seed_data"}:
                return False
            if not check_no_forbidden_keys(child):
                return False
    if isinstance(value, list):
        return all(check_no_forbidden_keys(item) for item in value)
    return True


def validate(_args):
    out_dir = env_path("TRAIN_CORPUS_DIR", DEFAULT_OUT)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")
    body = manifest_path.read_text(encoding="utf-8")
    if TOKEN_RE.search(body):
        raise SystemExit("manifest contains a Hugging Face token")
    manifest = json.loads(body)
    expected = {"train": 0, "val": 0, "holdout": 0}
    rows = 0
    for split, path, line_no, row in iter_jsonl(out_dir):
        validate_row(split, path, line_no, row, expected)
        rows += 1
    if rows == 0:
        raise SystemExit("no public-pretrain rows found")
    if expected != manifest.get("split_rows"):
        raise SystemExit("split row counts do not match manifest")
    report = {
        "schema": "lkjai-public-pretrain-validation",
        "status": "pass",
        "row_count": rows,
        "split_rows": expected,
        "manifest": str(manifest_path),
    }
    write_json(out_dir / "validation-report.json", report)
    print(json.dumps({"status": "pass", "rows": rows}))


def validate_row(split, path, line_no, row, expected):
    if not check_no_forbidden_keys(row):
        raise SystemExit(f"forbidden source field in {path}:{line_no}")
    if row.get("mode") != "pretrain" or row.get("language") != "en":
        raise SystemExit(f"bad row mode/language in {path}:{line_no}")
    text = row.get("text")
    meta = row.get("metadata", {})
    if not isinstance(text, str) or not text.strip():
        raise SystemExit(f"empty text in {path}:{line_no}")
    if meta.get("provenance") != "public-pretrain":
        raise SystemExit(f"bad provenance in {path}:{line_no}")
    if not meta.get("source_revision") or not meta.get("source_sha256"):
        raise SystemExit(f"missing source metadata in {path}:{line_no}")
    expected[split] += 1

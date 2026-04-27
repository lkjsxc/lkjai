import json

import pytest

import lkjai_train.public_pretrain as public_pretrain
from lkjai_train.paths import Paths
from lkjai_train.public_pretrain import prepare_public_pretrain, validate_public_pretrain_sources


def test_public_pretrain_sources_are_pinned_and_permissive(tmp_path):
    paths = Paths(str(tmp_path))
    manifest = validate_public_pretrain_sources(paths)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["sources"]
    assert {source["license"] for source in data["sources"]} == {"Apache-2.0"}
    assert all(source["language"] == "en" for source in data["sources"])
    assert all(len(source["revision"]) >= 20 for source in data["sources"])


def test_public_pretrain_rejects_bad_source(tmp_path, monkeypatch):
    paths = Paths(str(tmp_path))
    monkeypatch.setattr(public_pretrain, "public_pretrain_sources", lambda: [source("CC-BY-4.0", "abc123")])
    with pytest.raises(ValueError, match="disallowed license"):
        validate_public_pretrain_sources(paths)
    monkeypatch.setattr(public_pretrain, "public_pretrain_sources", lambda: [source("Apache-2.0", "main")])
    with pytest.raises(ValueError, match="pin an immutable revision"):
        validate_public_pretrain_sources(paths)


def test_prepare_public_pretrain_streams_local_jsonl(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    local = raw / "fixture.jsonl"
    text = " ".join(["This is a useful English educational paragraph about careful measurement."] * 20)
    local.write_text(json.dumps({"text": text}) + "\n" + json.dumps({"text": "too short"}) + "\n", encoding="utf-8")
    paths = Paths(str(tmp_path / "data"))
    monkeypatch.setenv("TRAIN_PUBLIC_DATA_DIR", str(raw))
    monkeypatch.setattr(public_pretrain, "public_pretrain_sources", lambda: [source("Apache-2.0", "abc123def456", local_file="fixture.jsonl")])
    out = prepare_public_pretrain(paths, target_tokens=50)
    train_files = sorted((out / "train").glob("*.jsonl"))
    assert train_files
    row = json.loads(train_files[0].read_text(encoding="utf-8").splitlines()[0])
    assert row["mode"] == "pretrain"
    assert row["metadata"]["provenance"] == "public-pretrain"
    report = json.loads((out / "validation-report.json").read_text(encoding="utf-8"))
    assert report["approx_tokens"] >= 50
    assert report["duplicate_rows"] == 0


def test_prepare_public_pretrain_rejects_partial_target(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "fixture.jsonl").write_text("", encoding="utf-8")
    paths = Paths(str(tmp_path / "data"))
    monkeypatch.setenv("TRAIN_PUBLIC_DATA_DIR", str(raw))
    monkeypatch.setattr(public_pretrain, "public_pretrain_sources", lambda: [source("Apache-2.0", "abc123def456")])
    with pytest.raises(RuntimeError, match="target"):
        prepare_public_pretrain(paths, target_tokens=50)
    assert (paths.public_pretrain / "validation-report.json").is_file()


def source(license_name: str, revision: str, local_file: str = "fixture.jsonl") -> dict:
    item = {
        "name": "fixture",
        "license": license_name,
        "source_url": "https://example.invalid/fixture",
        "revision": revision,
        "text_field": "text",
        "language": "en",
        "token_budget": 100,
    }
    if local_file:
        item["local_file"] = local_file
    return item

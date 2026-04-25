import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts" / "kimi_corpus" / "generate_kimi_corpus.py"
SCORER = ROOT / "scripts" / "kimi_corpus" / "score_corpus.py"
FAKE_KIMI = ROOT / "training" / "tests" / "fixtures" / "fake_kimi.py"
FAKE_KIMI_BAD = ROOT / "training" / "tests" / "fixtures" / "fake_kimi_bad.py"


def run(*args, cwd=ROOT):
    return subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def test_generator_and_scorer_help():
    assert "Generate resumable" in run(sys.executable, str(GENERATOR), "--help").stdout
    assert "Score Kimi" in run(sys.executable, str(SCORER), "--help").stdout


def test_fake_kimi_generates_valid_mixed_corpus(tmp_path):
    out = tmp_path / "corpus"
    run(
        sys.executable,
        str(GENERATOR),
        "--config",
        "configs/corpus/kimi_debug.yaml",
        "--target-tokens",
        "200",
        "--mode",
        "mixed",
        "--fake-kimi",
        str(FAKE_KIMI),
        "--output-dir",
        str(out),
        "--max-calls",
        "2",
        "--batch-documents",
        "3",
        "--run-dir",
        str(tmp_path / "runs"),
    )
    manifest = out / "manifest.jsonl"
    assert manifest.exists()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert all(row["validation_status"] == "valid" for row in rows)


def test_scorer_flags_malformed_duplicate_and_chat_contamination(tmp_path):
    path = tmp_path / "bad.jsonl"
    doc = {
        "id": "bad-1",
        "mode": "pretrain",
        "language": "en",
        "domain": "general",
        "difficulty": "introductory",
        "title": "Bad",
        "text": "User: hello\nAssistant: as an AI I cannot.\n" * 8,
        "metadata": {"source": "kimi_synthetic", "mode": "pretrain"},
    }
    path.write_text(json.dumps(doc) + "\n" + json.dumps(doc) + "\nnot json\n", encoding="utf-8")
    result = run(sys.executable, str(SCORER), str(path))
    summary = json.loads(result.stdout)
    assert summary["invalid_json_lines"]
    assert summary["duplicate_documents"] == 1
    assert summary["flag_counts"]["pretrain_chat_contamination"] >= 1
    assert summary["flag_counts"]["repeated_lines"] >= 1


def test_malformed_kimi_output_is_quarantined(tmp_path):
    out = tmp_path / "corpus"
    run(
        sys.executable,
        str(GENERATOR),
        "--config",
        "configs/corpus/kimi_debug.yaml",
        "--target-tokens",
        "200",
        "--mode",
        "pretrain",
        "--fake-kimi",
        str(FAKE_KIMI_BAD),
        "--output-dir",
        str(out),
        "--max-calls",
        "1",
        "--max-retries",
        "0",
        "--run-dir",
        str(tmp_path / "runs-bad"),
    )
    assert list((out / "quarantine" / "pretrain").glob("*.txt"))


def test_manifest_resume_appends_without_overwriting_completed_shards(tmp_path):
    out = tmp_path / "corpus"
    run(
        sys.executable,
        str(GENERATOR),
        "--config",
        "configs/corpus/kimi_debug.yaml",
        "--target-tokens",
        "200",
        "--mode",
        "pretrain",
        "--fake-kimi",
        str(FAKE_KIMI),
        "--output-dir",
        str(out),
        "--max-calls",
        "1",
        "--batch-documents",
        "2",
        "--run-dir",
        str(tmp_path / "runs-resume-1"),
    )
    first = out / "pretrain" / "train" / "shard_000001.jsonl"
    first_contents = first.read_text(encoding="utf-8")
    run(
        sys.executable,
        str(GENERATOR),
        "--config",
        "configs/corpus/kimi_debug.yaml",
        "--target-tokens",
        "400",
        "--mode",
        "pretrain",
        "--fake-kimi",
        str(FAKE_KIMI),
        "--output-dir",
        str(out),
        "--max-calls",
        "1",
        "--batch-documents",
        "2",
        "--resume",
        "--run-dir",
        str(tmp_path / "runs-resume-2"),
    )
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["shard_id"] for row in rows] == [1, 2]
    assert first.read_text(encoding="utf-8") == first_contents


def test_background_launch_dry_run():
    result = run("bash", "scripts/kimi_corpus/launch_background.sh", "--config", "configs/corpus/kimi_debug.yaml", "--target-tokens", "50000", "--dry-run")
    assert "dry-run command:" in result.stdout


def test_pretrain_and_sft_shapes_are_training_compatible(tmp_path):
    out = tmp_path / "corpus"
    run(
        sys.executable,
        str(GENERATOR),
        "--config",
        "configs/corpus/kimi_debug.yaml",
        "--target-tokens",
        "200",
        "--mode",
        "pretrain",
        "--fake-kimi",
        str(FAKE_KIMI),
        "--output-dir",
        str(out),
        "--max-calls",
        "1",
        "--batch-documents",
        "2",
        "--run-dir",
        str(tmp_path / "runs-pretrain"),
    )
    pretrain_row = json.loads(next((out / "pretrain" / "train").glob("*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
    assert "text" in pretrain_row
    run(
        sys.executable,
        str(GENERATOR),
        "--config",
        "configs/corpus/kimi_debug.yaml",
        "--target-tokens",
        "200",
        "--mode",
        "sft",
        "--fake-kimi",
        str(FAKE_KIMI),
        "--output-dir",
        str(out),
        "--max-calls",
        "1",
        "--batch-documents",
        "2",
        "--run-dir",
        str(tmp_path / "runs-sft"),
    )
    sft_row = json.loads(next((out / "sft" / "train").glob("*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
    assert {"messages", "tags", "meta"}.issubset(sft_row)

#!/usr/bin/env python3
import json
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "tools" / "benchmarks"))
    from run_support import load_train_report, summarize_train_report

    payload = {
        "schema_version": 1,
        "trainer_mode": "smoke",
        "model_kind": "dense",
        "optimizer_steps": 2,
        "microsteps": 2,
        "tokens_seen": 32,
        "elapsed_seconds": 0.25,
        "tokens_per_second": 128.0,
        "logits_checksum": "abc",
        "checkpoint_checksum": "def",
        "export_checksum": "123",
        "timings": {
            "batch_load": 0.01,
            "forward": 0.02,
            "backward": 0.03,
            "optimizer": 0.04,
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report = root / "runs" / "train-report.json"
        report.parent.mkdir()
        report.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load_train_report(root)
        summary = summarize_train_report(loaded)
        assert summary["schema_version"] == 1
        assert summary["trainer_mode"] == "smoke"
        assert summary["model_kind"] == "dense"
        assert summary["optimizer_steps"] == 2
        assert summary["median_step_seconds"] == 0.125
        assert summary["median_tokens_per_second"] == 128.0
        assert summary["logits_checksum"] == "abc"
        assert not (root / "runs" / "perf-steps.jsonl").exists()
        log = root / "trainer.log"
        report.unlink()
        log.write_text("noise\n" + json.dumps(payload) + "\n", encoding="utf-8")
        assert load_train_report(root, log)["logits_checksum"] == "abc"
        transformer = dict(payload)
        transformer.update(
            {
                "schema_version": 2,
                "trainer_mode": "train",
                "model_kind": "transformer",
                "layers": 1,
                "heads": 4,
                "hidden_size": 32,
                "ffn_size": 64,
            }
        )
        report.write_text(json.dumps(transformer), encoding="utf-8")
        summary = summarize_train_report(load_train_report(root))
        assert summary["schema_version"] == 2
        assert summary["model_kind"] == "transformer"
        assert summary["checkpoint_checksum"] == "def"


if __name__ == "__main__":
    main()

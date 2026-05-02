#!/usr/bin/env python3
import json
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "tools" / "benchmarks"))
    from run_support import (
        dense_promotion_errors,
        is_promotable_dense_summary,
        load_train_report,
        summarize_train_report,
        validate_dense_promotion_report,
    )

    payload = {
        "schema_version": 3,
        "trainer_mode": "smoke",
        "run_purpose": "accepted_training",
        "status": "success",
        "model_kind": "dense",
        "accepted_cuda_training": True,
        "implementation_status": "accepted",
        "loader_backend": "persistent_packed_cache_reader",
        "row_layout": "dense_physical_bxseq_masked_final_token",
        "matmul_plan_cache_enabled": True,
        "buffer_reuse_enabled": True,
        "timing_source": "cuda_events_with_boundary_sync",
        "optimizer_steps": 2,
        "microsteps": 2,
        "tokens_seen": 32,
        "initial_loss": 2.0,
        "loss": 1.5,
        "elapsed_seconds": 0.25,
        "tokens_per_second": 128.0,
        "logits_checksum": "abc",
        "checkpoint_checksum": "def",
        "export_checksum": "123",
        "logits_check": {
            "status": "pass",
            "reference_check": "pass",
            "max_abs_diff": 0.001,
            "tolerance": 0.01,
        },
        "timings": {
            "batch_load": 0.01,
            "h2d": 0.005,
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
        assert summary["schema_version"] == 3
        assert summary["trainer_mode"] == "smoke"
        assert summary["run_purpose"] == "accepted_training"
        assert summary["status"] == "success"
        assert summary["model_kind"] == "dense"
        assert summary["accepted_cuda_training"] is True
        assert summary["implementation_status"] == "accepted"
        assert summary["loader_backend"] == "persistent_packed_cache_reader"
        assert summary["matmul_plan_cache_enabled"] is True
        assert summary["optimizer_steps"] == 2
        assert summary["median_step_seconds"] == 0.125
        assert summary["median_tokens_per_second"] == 128.0
        assert summary["logits_checksum"] == "abc"
        validate_dense_promotion_report(loaded)
        promotable_row = dict(summary)
        promotable_row["returncode"] = 0
        assert is_promotable_dense_summary(promotable_row) is True
        assert not (root / "runs" / "perf-steps.jsonl").exists()
        log = root / "trainer.log"
        report.unlink()
        log.write_text("noise\n" + json.dumps(payload) + "\n", encoding="utf-8")
        assert load_train_report(root, log)["logits_checksum"] == "abc"
        transformer = dict(payload)
        transformer.update(
            {
                "schema_version": 3,
                "trainer_mode": "train",
                "model_kind": "transformer",
                "accepted_cuda_training": False,
                "implementation_status": "experimental",
                "layers": 1,
                "heads": 4,
                "hidden_size": 32,
                "ffn_size": 64,
            }
        )
        report.write_text(json.dumps(transformer), encoding="utf-8")
        summary = summarize_train_report(load_train_report(root))
        assert summary["schema_version"] == 3
        assert summary["model_kind"] == "transformer"
        assert summary["accepted_cuda_training"] is False
        assert summary["checkpoint_checksum"] == "def"
        transformer_row = dict(summary)
        transformer_row["returncode"] = 0
        assert is_promotable_dense_summary(transformer_row) is False

        failed = dict(payload)
        failed["status"] = "fail"
        failed_summary = summarize_train_report(failed)
        failed_summary["returncode"] = 2
        assert is_promotable_dense_summary(failed_summary) is False

        compatibility = dict(payload)
        compatibility["run_purpose"] = "bounded_compatibility_start_check"
        compatibility["limitations"] = ["bounded_compatibility_start_check"]
        compatibility_summary = summarize_train_report(compatibility)
        compatibility_summary["returncode"] = 0
        assert is_promotable_dense_summary(compatibility_summary) is False
        errors = dense_promotion_errors(compatibility)
        assert (
            "run_purpose bounded_compatibility_start_check is not promotable"
            in errors
        )

        missing_checksum = dict(payload)
        missing_checksum["export_checksum"] = ""
        errors = dense_promotion_errors(missing_checksum)
        assert "export_checksum must be present" in errors

        bad_logits = dict(payload)
        bad_logits["logits_check"] = dict(payload["logits_check"])
        bad_logits["logits_check"]["max_abs_diff"] = 0.02
        bad_logits["logits_check"]["tolerance"] = 0.01
        errors = dense_promotion_errors(bad_logits)
        assert "logits_check max_abs_diff exceeds tolerance" in errors
        missing_loader = dict(payload)
        missing_loader["loader_backend"] = ""
        errors = dense_promotion_errors(missing_loader)
        assert "loader_backend must be persistent_packed_cache_reader" in errors


if __name__ == "__main__":
    main()

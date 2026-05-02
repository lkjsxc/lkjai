#!/usr/bin/env python3
import sys
from pathlib import Path


def fixture_report() -> dict:
    return {
        "schema_version": 3,
        "trainer_mode": "train",
        "status": "success",
        "model_kind": "dense",
        "accepted_cuda_training": True,
        "implementation_status": "accepted",
        "forward_backend": "cuda_bf16_cublaslt",
        "backward_backend": "cuda_custom_or_gemm",
        "optimizer_backend": "cuda_adamw_fp32",
        "cuda_device_name": "Synthetic CUDA",
        "batch_size": 1,
        "seq_len": 16,
        "parameter_count": 16384,
        "optimizer_steps": 128,
        "start_step": 0,
        "initial_loss": 2.0,
        "loss": 1.0,
        "elapsed_seconds": 4.0,
        "tokens_per_second": 512.0,
        "checkpoint_path": "/app/data/perf-runs/run/case/repeat-01/checkpoints/latest",
        "checkpoint_checksum": "checkpoint-digest",
        "export_path": "/app/data/perf-runs/run/case/repeat-01/exports/model",
        "export_checksum": "export-digest",
        "logits_checksum": "logits-digest",
        "logits_check": {
            "status": "pass",
            "reference_check": "pass",
            "max_abs_diff": 0.001,
            "tolerance": 0.01,
        },
        "timings": {
            "batch_load": 0.1,
            "h2d": 0.2,
            "forward": 0.3,
            "backward": 0.4,
            "optimizer": 0.5,
            "checkpoint": 0.6,
            "export": 0.7,
        },
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "tools" / "benchmarks"))
    from dense_debug_support import promotion_summary
    from run_support import dense_promotion_errors, validate_dense_promotion_report

    report = fixture_report()
    validate_dense_promotion_report(report)
    summary = promotion_summary(
        "synthetic-run",
        "dense_debug_train_128",
        report,
        report["logits_check"],
        {"status": "success", "start_step": 128, "optimizer_steps": 129},
        5.0,
    )
    assert summary["promotion_status"] == "promoted"
    assert summary["hidden_size"] == 32
    assert summary["tokens_per_second"] == 512.0
    assert summary["h2d_fraction"] == 0.05
    assert summary["checksums"]["checkpoint"] == "checkpoint-digest"
    assert summary["resume_check"]["start_step"] == 128

    missing_reference = fixture_report()
    missing_reference["logits_check"] = dict(missing_reference["logits_check"])
    missing_reference["logits_check"]["reference_check"] = "not_requested"
    assert "logits_check.reference_check must be pass" in dense_promotion_errors(
        missing_reference
    )

    nondescending = fixture_report()
    nondescending["loss"] = nondescending["initial_loss"]
    assert "loss must be lower than initial_loss" in dense_promotion_errors(
        nondescending
    )


if __name__ == "__main__":
    main()

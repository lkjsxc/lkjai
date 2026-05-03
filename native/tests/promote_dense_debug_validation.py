#!/usr/bin/env python3
import sys
from pathlib import Path


def fixture_report() -> dict:
    return {
        "schema_version": 3,
        "trainer_mode": "train",
        "run_purpose": "dense_learning_control",
        "status": "success",
        "model_kind": "dense",
        "accepted_cuda_training": True,
        "implementation_status": "accepted",
        "loader_backend": "persistent_packed_cache_reader",
        "row_layout": "dense_physical_bxseq_masked_final_token",
        "matmul_plan_cache_enabled": True,
        "buffer_reuse_enabled": True,
        "timing_source": "cuda_events_with_boundary_sync",
        "forward_backend": "cuda_bf16_cublaslt",
        "backward_backend": "cuda_bf16_cublaslt_scatter",
        "backward_gemm_enabled": True,
        "embedding_grad_backend": "token_scatter_add_fp32",
        "dense_step_logits_bytes": 8192,
        "dense_step_grad_logits_bytes": 8192,
        "dense_step_d_hidden_bytes": 1024,
        "dense_logits_readback_bytes": 1024,
        "loss_kernel_backend": "block_row_softmax_fp32",
        "loss_readback_mode": "optimizer_step_deferred_pinned",
        "logits_readback_mode": "single_row_capture",
        "dense_stream_count": 2,
        "dense_batch_slot_count": 3,
        "copy_compute_overlap_enabled": True,
        "batch_staging_backend": "triple_slot_pinned_direct_read",
        "cublaslt_workspace_bytes": 4194304,
        "optimizer_backend": "cuda_adamw_fp32",
        "cuda_device_name": "Synthetic CUDA",
        "cuda_driver_version": 12080,
        "cuda_runtime_version": 12080,
        "cudnn_version": 901800,
        "cuda_device_count": 1,
        "cuda_device_index": 0,
        "cuda_total_global_memory": 8589934592,
        "cuda_sm_count": 46,
        "cuda_arch_flags": "86-real,86-virtual,120-real,120-virtual",
        "batch_size": 1,
        "seq_len": 16,
        "parameter_count": 16384,
        "optimizer_steps": 128,
        "start_step": 0,
        "initial_loss": 2.0,
        "loss": 1.0,
        "loss_samples": [
            {"step": 1, "loss": 2.0},
            {"step": 64, "loss": 1.4},
            {"step": 128, "loss": 1.0},
        ],
        "loss_sample_interval": 64,
        "best_loss": 1.0,
        "best_loss_step": 128,
        "loss_delta": 1.0,
        "loss_decrease_fraction": 0.5,
        "first_quarter_loss_mean": 2.0,
        "last_quarter_loss_mean": 1.0,
        "learning_status": "learning",
        "weight_changed": True,
        "weight_change": {"status": "pass"},
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
    assert summary["cuda_sm_count"] == 46
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

    unknown_purpose = fixture_report()
    unknown_purpose["run_purpose"] = "ad_hoc_smoke"
    assert (
        "run_purpose must be accepted_training or dense_learning_control"
        in dense_promotion_errors(unknown_purpose)
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import sys
from pathlib import Path


def accepted_report() -> dict:
    return {
        "schema_version": 3,
        "trainer_mode": "train",
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
        "optimizer_steps": 1024,
        "steps": 1024,
        "microsteps": 1024,
        "tokens_seen": 524288,
        "input_tokens": 524288,
        "loss_tokens": 520192,
        "initial_loss": 2.0,
        "loss": 1.0,
        "loss_samples": [
            {"step": 1, "loss": 2.0},
            {"step": 128, "loss": 1.85},
            {"step": 256, "loss": 1.70},
            {"step": 384, "loss": 1.55},
            {"step": 512, "loss": 1.42},
            {"step": 640, "loss": 1.30},
            {"step": 768, "loss": 1.22},
            {"step": 896, "loss": 1.12},
            {"step": 1024, "loss": 1.0},
        ],
        "loss_sample_interval": 128,
        "best_loss": 1.0,
        "best_loss_step": 1024,
        "loss_delta": 1.0,
        "loss_decrease_fraction": 0.5,
        "first_quarter_loss_mean": 1.925,
        "last_quarter_loss_mean": 1.06,
        "learning_status": "learning",
        "elapsed_seconds": 8.0,
        "tokens_per_second": 65536.0,
        "checkpoint_checksum": "checkpoint",
        "export_checksum": "export",
        "logits_checksum": "logits",
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


def accepted_summary(report: dict) -> dict:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "tools" / "benchmarks"))
    from run_support import summarize_train_report

    summary = summarize_train_report(report)
    summary.update(
        {
            "returncode": 0,
            "cache_metadata": {
                "row_count": 256,
                "source_digest": "source",
                "tokenizer_digest": "tokenizer",
                "config_digest": "config",
                "packed_data_checksum": "packed",
            },
            "checksums": {
                "checkpoint": "checkpoint",
                "export": "export",
                "logits": "logits",
            },
            "logits_reference": report["logits_check"],
            "infer": [
                {"status": "pass", "checksum": "logits"},
                {"status": "pass", "checksum": "logits"},
            ],
        }
    )
    return summary


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "tools" / "benchmarks"))
    from run_support import accepted_training_summary_errors, is_promotable_dense_summary
    from dense_accepted_training_support import token_accounting

    report = accepted_report()
    summary = accepted_summary(report)
    assert summary["cuda_driver_version"] == 12080
    assert summary["cuda_sm_count"] == 46
    assert accepted_training_summary_errors(summary) == []
    assert is_promotable_dense_summary(summary) is True

    accounting = token_accounting(report)
    assert accounting["valid"] is True
    assert accounting["non_loss_tokens"] == 4096

    bad_accounting = dict(summary)
    bad_accounting["loss_tokens"] = bad_accounting["tokens_seen"]
    assert "loss_tokens must be below tokens_seen" in accepted_training_summary_errors(
        bad_accounting
    )

    short_run = dict(summary)
    short_run["optimizer_steps"] = 256
    assert "optimizer_steps must be at least 1024" in accepted_training_summary_errors(
        short_run
    )


if __name__ == "__main__":
    main()

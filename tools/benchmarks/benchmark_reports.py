import math


PROMOTABLE_RUN_PURPOSES = {"accepted_training", "dense_learning_control"}


def _sample_loss(sample) -> float:
    if isinstance(sample, dict):
        return float(sample.get("loss"))
    return float(sample)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summarize_train_report(report: dict) -> dict:
    timings = report.get("timings", {})
    logits_check = report.get("logits_check", {})
    elapsed = float(report.get("elapsed_seconds", 0.0))
    steps = int(report.get("optimizer_steps", report.get("steps", 0)))
    step_seconds = elapsed / steps if steps > 0 else 0.0
    return {
        "schema_version": report.get("schema_version", 0),
        "trainer_mode": report.get("trainer_mode", report.get("mode", "")),
        "run_purpose": report.get("run_purpose", ""),
        "status": report.get("status", ""),
        "model_kind": report.get("model_kind", "dense"),
        "accepted_cuda_training": bool(report.get("accepted_cuda_training", False)),
        "implementation_status": report.get("implementation_status", ""),
        "loader_backend": report.get("loader_backend", ""),
        "row_layout": report.get("row_layout", ""),
        "matmul_plan_cache_enabled": bool(report.get("matmul_plan_cache_enabled", False)),
        "buffer_reuse_enabled": bool(report.get("buffer_reuse_enabled", False)),
        "timing_source": report.get("timing_source", ""),
        "optimizer_steps": steps,
        "microsteps": int(report.get("microsteps", 0)),
        "tokens_seen": int(report.get("tokens_seen", report.get("input_tokens", 0))),
        "initial_loss": float(report.get("initial_loss", 0.0)),
        "loss": float(report.get("loss", 0.0)),
        "loss_samples": report.get("loss_samples", []),
        "loss_sample_interval": int(report.get("loss_sample_interval", 0)),
        "best_loss": float(report.get("best_loss", report.get("loss", 0.0))),
        "best_loss_step": int(report.get("best_loss_step", 0)),
        "loss_delta": float(
            report.get(
                "loss_delta",
                float(report.get("initial_loss", 0.0))
                - float(report.get("loss", 0.0)),
            )
        ),
        "loss_decrease_fraction": float(
            report.get("loss_decrease_fraction", 0.0)
        ),
        "first_quarter_loss_mean": float(
            report.get("first_quarter_loss_mean", 0.0)
        ),
        "last_quarter_loss_mean": float(
            report.get("last_quarter_loss_mean", 0.0)
        ),
        "learning_status": report.get("learning_status", ""),
        "median_step_seconds": step_seconds,
        "median_tokens_per_second": float(report.get("tokens_per_second", 0.0)),
        "mean_loader_wait_seconds": float(timings.get("batch_load", 0.0)),
        "mean_h2d_seconds": float(timings.get("h2d", 0.0)),
        "mean_forward_seconds": float(timings.get("forward", 0.0)),
        "mean_backward_seconds": float(timings.get("backward", 0.0)),
        "mean_optimizer_seconds": float(timings.get("optimizer", 0.0)),
        "logits_checksum": report.get("logits_checksum", ""),
        "checkpoint_checksum": report.get("checkpoint_checksum", ""),
        "export_checksum": report.get("export_checksum", ""),
        "logits_check_status": logits_check.get("status", ""),
        "logits_reference_check": logits_check.get("reference_check", ""),
        "logits_max_abs_diff": float(logits_check.get("max_abs_diff", 0.0)),
        "logits_tolerance": float(logits_check.get("tolerance", 0.0)),
    }


def dense_promotion_errors(report: dict) -> list[str]:
    errors = []
    if report.get("schema_version") != 3:
        errors.append("schema_version must be 3")
    if report.get("model_kind") != "dense":
        errors.append("model_kind must be dense")
    if report.get("accepted_cuda_training") is not True:
        errors.append("accepted_cuda_training must be true")
    if report.get("implementation_status") != "accepted":
        errors.append("implementation_status must be accepted")
    if report.get("status") != "success":
        errors.append("status must be success")
    if report.get("run_purpose") == "bounded_compatibility_start_check":
        errors.append("run_purpose bounded_compatibility_start_check is not promotable")
    elif report.get("run_purpose") not in PROMOTABLE_RUN_PURPOSES:
        errors.append("run_purpose must be accepted_training or dense_learning_control")
    if report.get("loader_backend") != "persistent_packed_cache_reader":
        errors.append("loader_backend must be persistent_packed_cache_reader")
    if report.get("row_layout") != "dense_physical_bxseq_masked_final_token":
        errors.append("row_layout must be dense_physical_bxseq_masked_final_token")
    if report.get("matmul_plan_cache_enabled") is not True:
        errors.append("matmul_plan_cache_enabled must be true")
    if report.get("buffer_reuse_enabled") is not True:
        errors.append("buffer_reuse_enabled must be true")
    if report.get("timing_source") != "cuda_events_with_boundary_sync":
        errors.append("timing_source must be cuda_events_with_boundary_sync")
    try:
        initial_loss = float(report.get("initial_loss"))
        loss = float(report.get("loss"))
        if not math.isfinite(initial_loss) or not math.isfinite(loss):
            errors.append("loss and initial_loss must be finite")
        elif not loss < initial_loss:
            errors.append("loss must be lower than initial_loss")
    except (TypeError, ValueError):
        errors.append("loss and initial_loss must be numeric")
    samples = report.get("loss_samples", [])
    if samples:
        try:
            if not all(math.isfinite(_sample_loss(sample)) for sample in samples):
                errors.append("loss_samples must be finite")
        except (TypeError, ValueError):
            errors.append("loss_samples must be numeric")
    if not report.get("checkpoint_checksum"):
        errors.append("checkpoint_checksum must be present")
    if not report.get("export_checksum"):
        errors.append("export_checksum must be present")
    if not report.get("logits_checksum"):
        errors.append("logits_checksum must be present")
    timings = report.get("timings", {})
    try:
        if float(timings.get("h2d")) < 0.0:
            errors.append("timings.h2d must be non-negative")
    except (TypeError, ValueError):
        errors.append("timings.h2d must be numeric")
    try:
        if float(report.get("tokens_per_second")) <= 0.0:
            errors.append("tokens_per_second must be positive")
    except (TypeError, ValueError):
        errors.append("tokens_per_second must be numeric")
    logits_check = report.get("logits_check", {})
    if logits_check.get("status") != "pass":
        errors.append("logits_check.status must be pass")
    if logits_check.get("reference_check") != "pass":
        errors.append("logits_check.reference_check must be pass")
    try:
        max_abs_diff = float(logits_check.get("max_abs_diff"))
        tolerance = float(logits_check.get("tolerance"))
        if max_abs_diff > tolerance:
            errors.append("logits_check max_abs_diff exceeds tolerance")
    except (TypeError, ValueError):
        errors.append("logits_check max_abs_diff and tolerance must be numeric")
    return errors


def validate_dense_promotion_report(report: dict) -> None:
    errors = dense_promotion_errors(report)
    if errors:
        raise ValueError("; ".join(errors))


def is_promotable_dense_summary(row: dict) -> bool:
    try:
        checks = [
            row.get("returncode", 0) == 0,
            row.get("schema_version") == 3,
            row.get("model_kind") == "dense",
            row.get("accepted_cuda_training") is True,
            row.get("implementation_status") == "accepted",
            row.get("status") == "success",
            row.get("run_purpose", "") in PROMOTABLE_RUN_PURPOSES,
            row.get("loader_backend") == "persistent_packed_cache_reader",
            row.get("row_layout") == "dense_physical_bxseq_masked_final_token",
            row.get("matmul_plan_cache_enabled") is True,
            row.get("buffer_reuse_enabled") is True,
            row.get("timing_source") == "cuda_events_with_boundary_sync",
            float(row.get("loss", 0.0)) < float(row.get("initial_loss", 0.0)),
            _finite(row.get("loss", 0.0)),
            _finite(row.get("initial_loss", 0.0)),
            bool(row.get("checkpoint_checksum")),
            bool(row.get("export_checksum")),
            bool(row.get("logits_checksum")),
            float(row.get("mean_h2d_seconds", -1.0)) >= 0.0,
            float(row.get("median_tokens_per_second", 0.0)) > 0.0,
            row.get("logits_check_status") == "pass",
            row.get("logits_reference_check") == "pass",
            float(row.get("logits_max_abs_diff", 0.0))
            <= float(row.get("logits_tolerance", 0.0)),
        ]
        return all(checks)
    except (TypeError, ValueError):
        return False

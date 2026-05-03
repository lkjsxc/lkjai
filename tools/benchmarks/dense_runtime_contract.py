EXPECTED_DENSE_RUNTIME = {
    "loss_kernel_backend": "block_row_softmax_fp32",
    "loss_readback_mode": "optimizer_step_deferred_pinned",
    "logits_readback_mode": "single_row_capture",
    "dense_stream_count": 2,
    "dense_batch_slot_count": 3,
    "copy_compute_overlap_enabled": True,
    "batch_staging_backend": "triple_slot_pinned_direct_read",
}


def summarize_runtime_fields(report: dict) -> dict:
    summary = dict(EXPECTED_DENSE_RUNTIME)
    for key, default in EXPECTED_DENSE_RUNTIME.items():
        value = report.get(key, default if isinstance(default, bool) else "")
        summary[key] = bool(value) if isinstance(default, bool) else value
    summary["dense_stream_count"] = int(report.get("dense_stream_count", 0))
    summary["dense_batch_slot_count"] = int(report.get("dense_batch_slot_count", 0))
    summary["dense_logits_readback_bytes"] = int(
        report.get("dense_logits_readback_bytes", 0)
    )
    return summary


def dense_runtime_errors(report: dict) -> list[str]:
    errors = []
    for key, expected in EXPECTED_DENSE_RUNTIME.items():
        if report.get(key) != expected:
            errors.append(f"{key} must be {expected}")
    try:
        if int(report.get("dense_logits_readback_bytes", 0)) <= 0:
            errors.append("dense_logits_readback_bytes must be positive")
    except (TypeError, ValueError):
        errors.append("dense_logits_readback_bytes must be numeric")
    return errors


def dense_runtime_summary_checks(row: dict) -> bool:
    return not dense_runtime_errors(row)

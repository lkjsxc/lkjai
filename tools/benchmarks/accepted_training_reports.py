import math


def _sample_loss(sample) -> float:
    if isinstance(sample, dict):
        return float(sample.get("loss"))
    return float(sample)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def sampled_loss_values(samples: list) -> list[float]:
    return [_sample_loss(sample) for sample in samples]


def _quarter_window_decreased(row: dict) -> bool:
    try:
        return float(row.get("last_quarter_loss_mean")) < float(
            row.get("first_quarter_loss_mean")
        )
    except (TypeError, ValueError):
        return False


def _cache_field(row: dict, key: str):
    cache = row.get("cache_metadata", {})
    if isinstance(cache, dict) and key in cache:
        return cache.get(key)
    return row.get(f"cache_{key}")


def _checksum_field(row: dict, key: str):
    checksums = row.get("checksums", {})
    if isinstance(checksums, dict) and key in checksums:
        return checksums.get(key)
    return row.get(f"{key}_checksum")


def accepted_training_summary_errors(row: dict) -> list[str]:
    errors = []
    if row.get("run_purpose") != "accepted_training":
        errors.append("run_purpose must be accepted_training")
    if row.get("status") != "success":
        errors.append("status must be success")
    try:
        if int(row.get("optimizer_steps", 0)) < 1024:
            errors.append("optimizer_steps must be at least 1024")
    except (TypeError, ValueError):
        errors.append("optimizer_steps must be numeric")
    try:
        samples = sampled_loss_values(row.get("loss_samples", []))
        if len(samples) < 8:
            errors.append("loss_samples must contain at least 8 samples")
        if not all(math.isfinite(value) for value in samples):
            errors.append("loss_samples must be finite")
    except (TypeError, ValueError):
        errors.append("loss_samples must be numeric")
    if row.get("learning_status") != "learning":
        errors.append("learning_status must be learning")
    try:
        if float(row.get("loss_decrease_fraction", 0.0)) < 0.10:
            errors.append("loss_decrease_fraction must be at least 0.10")
    except (TypeError, ValueError):
        errors.append("loss_decrease_fraction must be numeric")
    if not _quarter_window_decreased(row):
        errors.append("last-quarter sampled mean must be below first-quarter sampled mean")
    _append_token_errors(row, errors)
    _append_cache_errors(row, errors)
    _append_checksum_errors(row, errors)
    _append_logits_errors(row, errors)
    _append_infer_errors(row, errors)
    _append_timing_errors(row, errors)
    for key in ("forward_backend", "backward_backend", "optimizer_backend", "cuda_device_name"):
        if not row.get(key):
            errors.append(f"{key} must be present")
    return errors


def _append_token_errors(row: dict, errors: list[str]) -> None:
    try:
        tokens_seen = int(row.get("tokens_seen", 0))
        loss_tokens = int(row.get("loss_tokens", 0))
        if tokens_seen <= 0:
            errors.append("tokens_seen must be positive")
        if loss_tokens <= 0:
            errors.append("loss_tokens must be positive")
        if tokens_seen > 0 and loss_tokens >= tokens_seen:
            errors.append("loss_tokens must be below tokens_seen")
    except (TypeError, ValueError):
        errors.append("tokens_seen and loss_tokens must be numeric")


def _append_cache_errors(row: dict, errors: list[str]) -> None:
    try:
        if int(_cache_field(row, "row_count") or 0) < 32:
            errors.append("cache row_count must be at least 32")
    except (TypeError, ValueError):
        errors.append("cache row_count must be numeric")
    for key in ("source_digest", "tokenizer_digest", "config_digest", "packed_data_checksum"):
        if not _cache_field(row, key):
            errors.append(f"cache {key} must be present")


def _append_checksum_errors(row: dict, errors: list[str]) -> None:
    for key in ("checkpoint", "export", "logits"):
        if not _checksum_field(row, key):
            errors.append(f"{key} checksum must be present")


def _append_logits_errors(row: dict, errors: list[str]) -> None:
    logits = row.get("logits_reference", row.get("logits_check", {}))
    if not isinstance(logits, dict):
        logits = {}
    if logits.get("status", row.get("logits_check_status")) != "pass":
        errors.append("BF16 logits reference check status must be pass")
    if logits.get("reference_check", row.get("logits_reference_check")) != "pass":
        errors.append("BF16 logits reference_check must be pass")
    try:
        tolerance = float(logits.get("tolerance", row.get("logits_tolerance", 0.0)))
        max_abs_diff = float(
            logits.get("max_abs_diff", row.get("logits_max_abs_diff", 0.0))
        )
        if tolerance != 0.01:
            errors.append("BF16 logits tolerance must remain 0.01")
        if max_abs_diff > tolerance:
            errors.append("BF16 logits max_abs_diff exceeds tolerance")
    except (TypeError, ValueError):
        errors.append("BF16 logits max_abs_diff and tolerance must be numeric")


def _append_infer_errors(row: dict, errors: list[str]) -> None:
    infer = row.get("infer", [])
    if not isinstance(infer, list) or len(infer) < 2:
        errors.append("dense infer must run twice")
        return
    first, second = infer[0], infer[1]
    if not isinstance(first, dict) or not isinstance(second, dict):
        errors.append("dense infer results must be objects")
    elif first.get("status") != "pass" or second.get("status") != "pass":
        errors.append("dense infer status must pass twice")
    elif not first.get("checksum") or first.get("checksum") != second.get("checksum"):
        errors.append("dense infer checksums must match")


def _append_timing_errors(row: dict, errors: list[str]) -> None:
    try:
        if float(row.get("median_tokens_per_second", row.get("tokens_per_second", 0.0))) <= 0.0:
            errors.append("throughput must be positive")
    except (TypeError, ValueError):
        errors.append("throughput must be numeric")
    for key in (
        "mean_loader_wait_seconds",
        "mean_h2d_seconds",
        "mean_forward_seconds",
        "mean_backward_seconds",
        "mean_optimizer_seconds",
    ):
        if key not in row:
            errors.append(f"{key} must be present")
        elif not _finite(row.get(key)):
            errors.append(f"{key} must be finite")

ACCEPTED_FORWARD = "cuda_bf16_full_decoder"
ACCEPTED_BACKWARD = "cuda_bf16_full_decoder"


def decoder_acceptance_errors(report: dict) -> list[str]:
    errors: list[str] = []
    expected = {
        "model_kind": "decoder",
        "accepted_cuda_training": True,
        "implementation_status": "accepted",
        "decoder_status": "accepted",
        "decoder_cuda_path": True,
        "decoder_cuda_slice": "full_decoder",
        "forward_backend": ACCEPTED_FORWARD,
        "backward_backend": ACCEPTED_BACKWARD,
        "optimizer_backend": "cuda_adamw_fp32",
        "matmul_backend": "cublaslt",
    }
    for key, value in expected.items():
        if report.get(key) != value:
            errors.append(f"{key} must be {value!r}")
    if report.get("attention_backend") in {"host_reference", "not_implemented", ""}:
        errors.append("attention_backend must be a CUDA decoder backend")
    if report.get("decoder_block_backend") in {
        "host_reference",
        "static_reference",
        "not_implemented",
        "",
    }:
        errors.append("decoder_block_backend must be CUDA-backed")
    if not report.get("decode_supported", False):
        errors.append("decode_supported must be true")
    if report.get("status") not in {None, "success"}:
        errors.append("status must be success")
    if report.get("loss_finite") is False:
        errors.append("loss_finite must not be false")
    return errors


def is_accepted_cuda_decoder(report: dict) -> bool:
    return not decoder_acceptance_errors(report)


def explain_decoder_acceptance(report: dict) -> str:
    errors = decoder_acceptance_errors(report)
    if not errors:
        return "accepted CUDA decoder backend present"
    return (
        "decoder full acceptance requires a complete CUDA BF16 decoder backend; "
        "P0 server-contract or partial CUDA slices are not sufficient: "
        + "; ".join(errors)
    )

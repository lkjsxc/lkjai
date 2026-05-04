#!/usr/bin/env python3
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "tools" / "benchmarks"))
    from decoder_acceptance import (
        decoder_acceptance_errors,
        is_accepted_cuda_decoder,
    )

    p0 = {
        "model_kind": "decoder",
        "accepted_cuda_training": False,
        "implementation_status": "experimental",
        "decoder_status": "experimental",
        "decoder_cuda_path": False,
        "forward_backend": "host_reference",
    }
    assert not is_accepted_cuda_decoder(p0)
    assert any("accepted_cuda_training" in e for e in decoder_acceptance_errors(p0))

    partial = dict(p0)
    partial.update(
        {
            "implementation_status": "partial_cuda",
            "decoder_status": "partial_cuda",
            "decoder_cuda_path": True,
            "decoder_cuda_slice": "embedding_lm_head",
            "forward_backend": "cuda_bf16_embedding_lm_head",
            "backward_backend": "cuda_bf16_embedding_lm_head",
            "optimizer_backend": "cuda_adamw_fp32",
            "matmul_backend": "cublaslt",
            "attention_backend": "not_implemented",
            "decoder_block_backend": "static_reference",
            "decode_supported": True,
        }
    )
    assert not is_accepted_cuda_decoder(partial)
    assert any("decoder_cuda_slice" in e for e in decoder_acceptance_errors(partial))

    accepted = dict(partial)
    accepted.update(
        {
            "accepted_cuda_training": True,
            "implementation_status": "accepted",
            "decoder_status": "accepted",
            "decoder_cuda_slice": "full_decoder",
            "forward_backend": "cuda_bf16_full_decoder",
            "backward_backend": "cuda_bf16_full_decoder",
            "attention_backend": "cuda_sdpa",
            "decoder_block_backend": "cuda_bf16_full_decoder",
            "decode_supported": True,
            "status": "success",
        }
    )
    assert is_accepted_cuda_decoder(accepted)


if __name__ == "__main__":
    main()

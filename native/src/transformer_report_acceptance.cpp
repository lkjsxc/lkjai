#include "transformer_report_acceptance.hpp"

#include "decoder_decode.hpp"

namespace lkjai {

bool transformer_report_accepted_decoder(const TransformerTrainReport& r) {
  bool accepted_attention = r.attention_backend == "cuda_causal_gqa_bf16_reference" ||
                            r.attention_backend == "cudnn_sdpa";
  return r.model_kind == "decoder" && r.implementation_status == "accepted" &&
         accepted_attention && r.decoder_cuda_path &&
         r.decoder_cuda_slice == "full_decoder" &&
         r.decoder_block_backend == "cuda_full_decoder" &&
         r.forward_backend == "cuda_full_decoder" &&
         r.backward_backend == "cuda_full_decoder" &&
         r.decoder_backward_backend == "cuda_full_decoder" &&
         r.embedding_tying == "tok_embeddings:lm_head" &&
         r.kv_cache_backend == kDecoderAcceptedKvCacheBackend &&
         r.decode_backend == kDecoderAcceptedDecodeBackend;
}

std::vector<std::string> transformer_report_limitations(
    const TransformerTrainReport& r, bool accepted_decoder) {
  std::vector<std::string> out;
  if (r.run_purpose == "bounded_diagnostic_start_check") {
    out.push_back("bounded_diagnostic_start_check");
  }
  if (accepted_decoder) return out;
  out.push_back("experimental_not_accepted_cuda_training");
  out.push_back(r.decoder_cuda_path ? "partial_cuda_decoder_slice"
                                    : "host_reference_forward");
  out.push_back(r.decoder_cuda_path ? "decoder_forward_partial"
                                    : "host_surrogate_backward");
  if (r.forward_backend != "cuda_full_decoder") {
    out.push_back("full_forward_not_accepted");
  }
  if (r.backward_backend != "cuda_full_decoder") {
    out.push_back("full_backward_not_accepted");
  }
  if (r.attention_backend == "not_implemented") {
    out.push_back("attention_not_implemented");
  }
  if (r.decoder_backward_backend == "not_implemented") {
    out.push_back("decoder_backward_not_implemented");
  }
  if (r.kv_cache_backend == "none") out.push_back("kv_cache_not_implemented");
  if (!r.decode_supported) out.push_back("autoregressive_decode_unsupported");
  return out;
}

}  // namespace lkjai

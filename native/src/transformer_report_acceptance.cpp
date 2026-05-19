#include "transformer_report_acceptance.hpp"

#include <cmath>

#include "decoder_decode.hpp"
#include "json_min.hpp"

namespace lkjai {
namespace {

bool positive_block_weight_evidence(const TransformerTrainReport& r) {
  const auto& w = r.decoder_weight_change;
  return w.non_embedding.max_abs_delta > 0.0 &&
         w.non_embedding.changed_tensors > 0 &&
         w.decoder_block.max_abs_delta > 0.0 &&
         w.decoder_block.changed_tensors > 0;
}

bool accepted_attention_backend(const TransformerTrainReport& r) {
  return r.attention_backend == kDecoderAcceptedAttentionBackend;
}

bool rejected_diagnostic_backend(const TransformerTrainReport& r) {
  return r.decoder_backward_backend.find("diagnostic") != std::string::npos ||
         r.decoder_gradient_source.find("diagnostic") != std::string::npos ||
         r.attention_backend == kDecoderReferenceAttentionBackend ||
         r.decoder_gradient_source == "host_reference" ||
         r.decoder_backward_backend == "host_reference";
}

bool accepted_decode_support(const TransformerTrainReport& r) {
  return r.decode_supported &&
         r.kv_cache_backend == kDecoderAcceptedKvCacheBackend &&
         r.decode_backend == kDecoderAcceptedDecodeBackend &&
         r.kv_cache_prefill_allocated_bytes > 0 &&
         r.kv_cache_steady_state_token_allocations == 0;
}

bool accepted_runtime_evidence(const TransformerTrainReport& r) {
  const auto& e = r.decoder_runtime_evidence;
  return e.cudnn_sdpa_forward_count > 0 && e.cudnn_sdpa_backward_count > 0 &&
         e.attention_reference_forward_count == 0 &&
         e.attention_reference_backward_count == 0 &&
         e.cudnn_sdpa_workspace_bytes > 0 &&
         r.decoder_parity_sample_count > 0 &&
         r.decoder_parity_failure_count == 0;
}

bool accepted_40m_3070_shape(const TransformerTrainReport& r) {
  return r.config_path.filename() == "decoder_40m_bf16_3070.json" &&
         r.train_config_path.filename() == "decoder_2h_40m_3070.json" &&
         r.target_seconds >= 7200 && r.seq_len == 1024 && r.context == 1024 &&
         r.layers == 10 && r.hidden_size == 576 && r.heads == 8 &&
         r.kv_heads == 2 && r.head_dim == 72 && r.ffn_size == 1536;
}

}  // namespace

bool transformer_report_shape_accepted_decoder(const TransformerTrainReport& r) {
  return r.model_kind == "decoder" && r.implementation_status == "accepted" &&
         accepted_attention_backend(r) && r.decoder_cuda_path &&
         r.decoder_cuda_slice == "full_decoder" &&
         r.decoder_block_backend == "cuda_full_decoder" &&
         r.forward_backend == "cuda_full_decoder" &&
         r.backward_backend == "cuda_full_decoder" &&
         r.decoder_backward_backend == "cuda_full_decoder" &&
         r.decoder_gradient_source == "cuda_device" &&
         !rejected_diagnostic_backend(r) &&
         accepted_runtime_evidence(r) &&
         accepted_decode_support(r) && r.logits_check_passed &&
         std::isfinite(r.loss) && r.steps > 0 && r.loss_tokens > 0 &&
         r.trainable_weight_changed && r.non_embedding_weight_changed &&
         r.decoder_block_weight_changed && positive_block_weight_evidence(r) &&
         r.embedding_tying == "tok_embeddings:lm_head" &&
         accepted_40m_3070_shape(r);
}

bool transformer_report_accepted_decoder(const TransformerTrainReport& r) { return transformer_report_shape_accepted_decoder(r); }

std::vector<std::string> transformer_report_limitations(
    const TransformerTrainReport& r, bool accepted_decoder) {
  std::vector<std::string> out;
  if (r.run_purpose == "bounded_diagnostic_start_check") out.push_back("bounded_diagnostic_start_check");
  if (accepted_decoder) return out;
  out.push_back("experimental_not_accepted_cuda_training");
  out.push_back(r.decoder_cuda_path ? "partial_cuda_decoder_slice" : "host_reference_forward");
  out.push_back(r.decoder_cuda_path ? "decoder_forward_partial" : "host_surrogate_backward");
  if (r.forward_backend != "cuda_full_decoder") out.push_back("full_forward_not_accepted");
  if (r.backward_backend != "cuda_full_decoder") out.push_back("full_backward_not_accepted");
  if (r.attention_backend == "not_implemented")
    out.push_back("attention_not_implemented");
  if (r.decoder_backward_backend == "not_implemented")
    out.push_back("decoder_backward_not_implemented");
  if (r.decoder_backward_backend.find("diagnostic") != std::string::npos) out.push_back("decoder_backward_diagnostic_synthetic");
  if (r.decoder_gradient_source.find("diagnostic") != std::string::npos) out.push_back("decoder_gradients_device_diagnostic");
  if (r.decoder_gradient_source != "cuda_device") out.push_back("decoder_gradients_not_device_origin");
  if (r.model_kind == "decoder" && !r.decoder_block_weight_changed)
    out.push_back("decoder_block_weights_not_updated");
  if (r.model_kind == "decoder" && r.decoder_cuda_slice != "full_decoder")
    out.push_back("decoder_block_optimizer_not_implemented");
  if (r.kv_cache_backend == "none") out.push_back("kv_cache_not_implemented");
  if (!accepted_decode_support(r)) out.push_back("autoregressive_decode_unsupported");
  return out;
}

}  // namespace lkjai

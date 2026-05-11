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
  return r.attention_backend == "cuda_causal_gqa_bf16_reference" ||
         r.attention_backend == "cudnn_sdpa";
}

bool accepted_decode_support(const TransformerTrainReport& r) {
  return r.decode_supported &&
         r.kv_cache_backend == kDecoderAcceptedKvCacheBackend &&
         r.decode_backend == kDecoderAcceptedDecodeBackend &&
         r.kv_cache_prefill_allocated_bytes > 0 &&
         r.kv_cache_steady_state_token_allocations == 0;
}

double json_double_after(std::string_view text, std::string_view needle) {
  auto pos = text.find(needle);
  if (pos == std::string_view::npos) return 0.0;
  pos += needle.size();
  try {
    return std::stod(std::string(text.substr(pos)));
  } catch (...) {
    return 0.0;
  }
}

bool require_artifact(const std::filesystem::path& dir, std::string_view label,
                      std::string* error) {
  if (std::filesystem::is_regular_file(dir / "manifest.json")) return true;
  *error = std::string(label) + " artifact manifest missing";
  return false;
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
         accepted_decode_support(r) && r.logits_check_passed &&
         std::isfinite(r.loss) && r.steps > 0 && r.loss_tokens > 0 &&
         r.trainable_weight_changed && r.non_embedding_weight_changed &&
         r.decoder_block_weight_changed && positive_block_weight_evidence(r) &&
         r.embedding_tying == "tok_embeddings:lm_head";
}

bool transformer_report_accepted_decoder(const TransformerTrainReport& r) {
  return transformer_report_shape_accepted_decoder(r);
}

bool transformer_emitted_decoder_evidence_accepted(
    const std::filesystem::path& train_report, std::string* error) {
  auto body = read_text(train_report);
  if (body.empty()) {
    *error = "missing train-report.json";
    return false;
  }
  if (!contains_json_string(body, "status", "success") ||
      !contains_json_string(body, "model_kind", "decoder") ||
      !contains_json_string(body, "implementation_status", "accepted")) {
    *error = "train report is not successful accepted decoder evidence";
    return false;
  }
  if (!json_bool_value(body, "accepted_cuda_training", false) ||
      !json_bool_value(body, "logits_check_passed", false) ||
      !json_bool_value(body, "decode_supported", false)) {
    *error = "train report missing required accepted booleans";
    return false;
  }
  if (json_int_value(body, "optimizer_steps", 0) <= 0 ||
      json_int_value(body, "loss_tokens", 0) <= 0 ||
      !json_bool_value(body, "loss_finite", false)) {
    *error = "train report missing positive step/loss evidence";
    return false;
  }
  if (!json_bool_value(body, "trainable_weight_changed", false) ||
      !json_bool_value(body, "non_embedding_weight_changed", false) ||
      !json_bool_value(body, "decoder_block_weight_changed", false)) {
    *error = "train report missing decoder weight-change booleans";
    return false;
  }
  if (json_double_after(body, "\"non_embedding\":{\"max_abs_delta\":") <= 0.0 ||
      json_double_after(body, "\"decoder_block\":{\"max_abs_delta\":") <= 0.0) {
    *error = "train report missing positive decoder quantitative deltas";
    return false;
  }
  if (!contains_json_string(body, "decode_backend", kDecoderAcceptedDecodeBackend) ||
      !contains_json_string(body, "kv_cache_backend",
                            kDecoderAcceptedKvCacheBackend)) {
    *error = "train report missing accepted decode backends";
    return false;
  }
  if (json_int_value(body, "kv_cache_prefill_allocated_bytes", 0) <= 0 ||
      json_int_value(body, "kv_cache_steady_state_token_allocations", -1) != 0) {
    *error = "train report missing KV allocation accounting";
    return false;
  }
  if (!contains_json_string(body, "status", "pass")) {
    *error = "train report missing passing logits status";
    return false;
  }
  return require_artifact(json_first_string(body, "checkpoint_path"),
                          "checkpoint", error) &&
         require_artifact(json_first_string(body, "export_path"), "export",
                          error) &&
         require_artifact(json_first_string(body, "served_path"), "served",
                          error);
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
  if (r.model_kind == "decoder" && !r.decoder_block_weight_changed) {
    out.push_back("decoder_block_weights_not_updated");
  }
  if (r.model_kind == "decoder" && r.decoder_cuda_slice != "full_decoder") {
    out.push_back("decoder_block_optimizer_not_implemented");
  }
  if (r.kv_cache_backend == "none") out.push_back("kv_cache_not_implemented");
  if (!accepted_decode_support(r)) {
    out.push_back("autoregressive_decode_unsupported");
  }
  return out;
}

}  // namespace lkjai

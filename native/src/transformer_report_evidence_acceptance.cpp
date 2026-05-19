#include "transformer_report_acceptance.hpp"

#include "decoder_decode.hpp"
#include "json_min.hpp"

namespace lkjai {
namespace {

double json_double_after(std::string_view text, std::string_view needle) {
  auto pos = text.find(needle);
  if (pos == std::string_view::npos) return 0.0;
  try {
    return std::stod(std::string(text.substr(pos + needle.size())));
  } catch (...) {
    return 0.0;
  }
}

bool route_report_shape_ok(std::string_view body) {
  return json_int_value(body, "target_seconds", 0) >= 7200 &&
         json_int_value(body, "seq_len", 0) == 1024 &&
         json_int_value(body, "context", 0) == 1024 &&
         json_int_value(body, "layers", 0) == 10 &&
         json_int_value(body, "hidden_size", 0) == 576 &&
         json_int_value(body, "heads", 0) == 8 &&
         json_int_value(body, "kv_heads", 0) == 2 &&
         json_int_value(body, "head_dim", 0) == 72 &&
         json_int_value(body, "ffn_size", 0) == 1536;
}

bool require_artifact(const std::filesystem::path& dir, std::string_view label,
                      std::string* error) {
  if (std::filesystem::is_regular_file(dir / "manifest.json")) return true;
  *error = std::string(label) + " artifact manifest missing";
  return false;
}

bool report_booleans_ok(std::string_view body, std::string* error) {
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
  return true;
}

bool report_training_ok(std::string_view body, std::string* error) {
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
  return true;
}

bool report_backend_ok(std::string_view body, std::string* error) {
  if (!contains_json_string(body, "attention_backend",
                            kDecoderAcceptedAttentionBackend) ||
      !contains_json_string(body, "decoder_cuda_slice", "full_decoder") ||
      !contains_json_string(body, "forward_backend", "cuda_full_decoder") ||
      !contains_json_string(body, "backward_backend", "cuda_full_decoder") ||
      !contains_json_string(body, "decoder_backward_backend",
                            "cuda_full_decoder") ||
      !contains_json_string(body, "decoder_gradient_source", "cuda_device")) {
    *error = "train report missing accepted CUDA decoder backends";
    return false;
  }
  auto backward = json_first_string(body, "decoder_backward_backend");
  auto gradient = json_first_string(body, "decoder_gradient_source");
  auto attention = json_first_string(body, "attention_backend");
  auto decode = json_first_string(body, "decode_backend");
  if (backward == "cuda_diagnostic_synthetic" ||
      gradient == "cuda_device_diagnostic" || gradient == "host_reference" ||
      attention == kDecoderReferenceAttentionBackend ||
      decode == kDecoderPartialDecodeBackend ||
      decode == kDecoderRuntimePartialDecodeBackend) {
    *error = "train report contains diagnostic or partial decoder evidence";
    return false;
  }
  return true;
}

bool report_decode_ok(std::string_view body, std::string* error) {
  if (!contains_json_string(body, "decode_backend",
                            kDecoderAcceptedDecodeBackend) ||
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
  return true;
}

bool report_runtime_ok(std::string_view body, std::string* error) {
  if (json_int_value(body, "cudnn_sdpa_forward_count", 0) <= 0 ||
      json_int_value(body, "cudnn_sdpa_backward_count", 0) <= 0 ||
      json_int_value(body, "attention_reference_forward_count", 1) != 0 ||
      json_int_value(body, "attention_reference_backward_count", 1) != 0 ||
      json_int_value(body, "cudnn_sdpa_workspace_bytes", 0) <= 0) {
    *error = "train report missing cuDNN SDPA runtime evidence";
    return false;
  }
  if (json_int_value(body, "decoder_parity_sample_count", 0) <= 0 ||
      json_int_value(body, "decoder_parity_failure_count", 1) != 0) {
    *error = "train report missing passing decoder parity evidence";
    return false;
  }
  return true;
}

}  // namespace

bool transformer_emitted_decoder_evidence_accepted(
    const std::filesystem::path& train_report, std::string* error) {
  auto body = read_text(train_report);
  if (body.empty()) {
    *error = "missing train-report.json";
    return false;
  }
  if (!report_booleans_ok(body, error) || !report_training_ok(body, error) ||
      !report_backend_ok(body, error) || !report_decode_ok(body, error) ||
      !report_runtime_ok(body, error)) {
    return false;
  }
  if (!contains_json_string(body, "status", "pass")) {
    *error = "train report missing passing logits status";
    return false;
  }
  if (!route_report_shape_ok(body)) {
    *error = "train report is not the 40M RTX 3070 acceptance shape";
    return false;
  }
  auto device = json_first_string(body, "cuda_device_name");
  if (device.find("RTX 3070") == std::string::npos) {
    *error = "train report is not RTX 3070 evidence";
    return false;
  }
  return require_artifact(json_first_string(body, "checkpoint_path"),
                          "checkpoint", error) &&
         require_artifact(json_first_string(body, "export_path"), "export",
                          error) &&
         require_artifact(json_first_string(body, "served_path"), "served",
                          error);
}

}  // namespace lkjai

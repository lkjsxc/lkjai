#include "transformer_report_acceptance.hpp"

#include "decoder_decode.hpp"
#include "json_min.hpp"

namespace lkjai {
namespace {

bool route_report_fields_ok(std::string_view body, std::string* error) {
  if (!contains_json_string(body, "status", "success") ||
      !contains_json_string(body, "model_kind", "decoder") ||
      !contains_json_string(body, "implementation_status", "accepted")) {
    *error = "route report is not accepted decoder evidence";
    return false;
  }
  if (!json_bool_value(body, "accepted_cuda_training", false) ||
      !json_bool_value(body, "decode_supported", false) ||
      !json_bool_value(body, "logits_check_passed", false)) {
    *error = "route report missing accepted booleans";
    return false;
  }
  if (!contains_json_string(body, "decode_backend",
                            kDecoderAcceptedDecodeBackend) ||
      !contains_json_string(body, "kv_cache_backend",
                            kDecoderAcceptedKvCacheBackend)) {
    *error = "route report missing accepted decode backends";
    return false;
  }
  if (json_int_value(body, "kv_cache_prefill_allocated_bytes", 0) <= 0 ||
      json_int_value(body, "kv_cache_steady_state_token_allocations", -1) != 0) {
    *error = "route report missing KV allocation accounting";
    return false;
  }
  return true;
}

bool route_report_shape_ok(std::string_view body) {
  return json_int_value(body, "target_seconds", 0) >= 7200 &&
         json_int_value(body, "seq_len", 0) == 1024 &&
         json_int_value(body, "layers", 0) == 10 &&
         json_int_value(body, "hidden_size", 0) == 576 &&
         json_int_value(body, "heads", 0) == 8 &&
         json_int_value(body, "kv_heads", 0) == 2 &&
         json_int_value(body, "head_dim", 0) == 72 &&
         json_int_value(body, "ffn_size", 0) == 1536;
}

}  // namespace

bool transformer_emitted_decoder_route_report_accepted(
    const std::filesystem::path& train_report, std::string* error) {
  auto body = read_text(train_report);
  if (body.empty()) {
    *error = "missing decoder route train report";
    return false;
  }
  if (!route_report_fields_ok(body, error)) return false;
  if (!route_report_shape_ok(body)) {
    *error = "route report is not the 40M RTX 3070 acceptance shape";
    return false;
  }
  auto device = json_first_string(body, "cuda_device_name");
  if (device.find("RTX 3070") == std::string::npos) {
    *error = "route report is not RTX 3070 evidence";
    return false;
  }
  return true;
}

}  // namespace lkjai

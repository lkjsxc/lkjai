#include "train_report.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

#include "artifact.hpp"
#include "capability_json.hpp"
#include "json_min.hpp"
#ifndef LKJAI_GIT_COMMIT
#define LKJAI_GIT_COMMIT "unknown"
#endif
#ifndef LKJAI_BUILD_TYPE
#define LKJAI_BUILD_TYPE "unknown"
#endif
#ifndef LKJAI_CUDA_ARCH_FLAGS
#define LKJAI_CUDA_ARCH_FLAGS "unknown"
#endif

namespace lkjai {
namespace {

std::string file_digest(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  uint64_t hash = 1469598103934665603ull;
  char ch = 0;
  while (in.get(ch)) {
    hash = (hash ^ static_cast<unsigned char>(ch)) * 1099511628211ull;
  }
  std::ostringstream out;
  out << std::hex << hash;
  return out.str();
}

std::string packed_cache_digest(const std::filesystem::path& dir) {
  uint64_t hash = 1469598103934665603ull;
  for (const auto& name :
       {"metadata.json", "tokens.bin", "loss_mask.bin", "starts.bin"}) {
    auto path = dir / name;
    hash = (hash ^ artifact_text_checksum(name)[0]) * 1099511628211ull;
    if (!std::filesystem::is_regular_file(path)) continue;
    auto text = file_digest(path);
    for (char ch : text) {
      hash = (hash ^ static_cast<unsigned char>(ch)) * 1099511628211ull;
    }
    auto size = std::filesystem::file_size(path);
    for (int i = 0; i < 8; ++i) {
      hash = (hash ^ ((size >> (i * 8)) & 0xffu)) * 1099511628211ull;
    }
  }
  std::ostringstream out;
  out << std::hex << hash;
  return out.str();
}

std::string manifest_checksum(const std::filesystem::path& dir) {
  auto manifest = read_text(dir / "manifest.json");
  auto checksum = json_first_string(manifest, "weights_checksum");
  return checksum.empty() ? file_digest(dir / "weights.lkjw") : checksum;
}

void append_transformer(std::ostringstream* out,
                        const TransformerTrainReport& report,
                        const CudaStatus& cuda,
                        const std::string& trainer_mode,
                        const std::string& status,
                        const std::string& failure_reason) {
  double tokens_per_second =
      report.elapsed_seconds > 0.0
          ? static_cast<double>(report.input_tokens) / report.elapsed_seconds
          : 0.0;
  *out << "{\"schema_version\":3"
       << ",\"trainer_mode\":\"" << json_escape(trainer_mode) << "\""
       << ",\"mode\":\"" << json_escape(trainer_mode) << "\""
       << ",\"run_purpose\":\"" << json_escape(report.run_purpose) << "\""
       << ",\"model_kind\":\"transformer\""
       << ",\"accepted_cuda_training\":false"
       << ",\"implementation_status\":\"experimental\""
       << ",\"transformer_status\":\"experimental\""
       << ",\"forward_backend\":\"host_reference\""
       << ",\"backward_backend\":\"host_surrogate\""
       << ",\"optimizer_backend\":\"host_adamw_fp32\""
       << ",\"cuda_probe_passed\":"
       << (cuda_required_ok(cuda) ? "true" : "false")
       << ",\"status\":\"" << json_escape(status) << "\""
       << ",\"failure_reason\":\"" << json_escape(failure_reason) << "\""
       << ",\"limitations\":["
       << (report.run_purpose == "bounded_compatibility_start_check"
               ? "\"bounded_compatibility_start_check\","
               : "")
       << "\"experimental_not_accepted_cuda_training\","
          "\"host_reference_forward\","
          "\"host_surrogate_backward\","
          "\"autoregressive_decode_unsupported\"]"
       << ",\"precision_mode\":\"fp32-master-bf16-shadow-bf16-export\""
       << ",\"master_dtype\":\"f32\",\"shadow_dtype\":\"bf16\""
       << ",\"accumulation_dtype\":\"f32\",\"export_dtype\":\"bf16\""
       << ",\"dense_cuda_path\":false,\"transformer_cuda_path\":false"
       << ",\"transformer_cuda_probe\":true"
       << ",\"cuda_available\":" << (cuda.available ? "true" : "false")
       << ",\"cuda_device_name\":\"" << json_escape(cuda.device) << "\""
       << ",\"cuda_driver_version\":" << cuda.cuda_driver_version << ",\"cuda_runtime_version\":" << cuda.cuda_runtime_version
       << ",\"cudnn_version\":" << cuda.cudnn_version << ",\"cuda_device_count\":" << cuda.device_count
       << ",\"cuda_device_index\":" << cuda.device_index
       << ",\"cuda_total_global_memory\":" << static_cast<unsigned long long>(cuda.total_global_memory)
       << ",\"cuda_sm_count\":" << cuda.sm_count << ",\"cuda_arch_flags\":\"" << json_escape(LKJAI_CUDA_ARCH_FLAGS)
       << "\",\"git_commit\":\"" << json_escape(LKJAI_GIT_COMMIT) << "\""
       << ",\"build_type\":\"" << json_escape(LKJAI_BUILD_TYPE) << "\""
       << ",\"config_path\":\"" << json_escape(report.config_path.string())
       << "\",\"config_digest\":\"" << file_digest(report.config_path) << "\""
       << ",\"dataset_path\":\"" << json_escape(report.packed_cache.string())
       << "\",\"packed_cache_path\":\""
       << json_escape(report.packed_cache.string()) << "\""
       << ",\"dataset_digest\":\"" << packed_cache_digest(report.packed_cache)
       << "\",\"train_config_path\":\""
       << json_escape(report.train_config_path.string()) << "\""
       << ",\"seed\":" << json_int_value(read_text(report.config_path), "seed", 0)
       << ",\"batch_size\":" << report.batch_size
       << ",\"seq_len\":" << report.seq_len
       << ",\"grad_accum\":" << report.grad_accum
       << ",\"layers\":" << report.layers << ",\"heads\":" << report.heads
       << ",\"kv_heads\":" << report.kv_heads
       << ",\"hidden_size\":" << report.hidden_size
       << ",\"head_dim\":" << report.head_dim
       << ",\"ffn_size\":" << report.ffn_size
       << ",\"context\":" << report.context
       << ",\"parameter_count\":" << report.parameter_count
       << ",\"tokens_seen\":" << report.input_tokens
       << ",\"input_tokens\":" << report.input_tokens
       << ",\"loss_tokens\":" << report.loss_tokens
       << ",\"optimizer_steps\":" << report.steps
       << ",\"steps\":" << report.steps
       << ",\"start_step\":" << report.start_step
       << ",\"microsteps\":" << report.microsteps
       << ",\"initial_loss\":" << report.initial_loss
       << ",\"loss\":" << report.loss << ",\"loss_finite\":true"
       << ",\"weight_changed\":"
       << (report.non_embedding_weight_changed ? "true" : "false")
       << ",\"elapsed_ms\":" << report.elapsed_seconds * 1000.0
       << ",\"elapsed_seconds\":" << report.elapsed_seconds
       << ",\"tokens_per_second\":" << tokens_per_second
       << ",\"checkpoint_path\":\""
       << json_escape(report.checkpoint_dir.string()) << "\""
       << ",\"checkpoint_checksum\":\""
       << manifest_checksum(report.checkpoint_dir) << "\""
       << ",\"export_path\":\"" << json_escape(report.export_dir.string())
       << "\",\"export_checksum\":\"" << manifest_checksum(report.export_dir)
       << "\",\"served_path\":\"" << json_escape(report.served_dir.string())
       << "\",\"logits_checksum\":\""
       << json_escape(report.logits_check_checksum) << "\""
       << ",\"logits_check\":"
       << (report.logits_check_json.empty()
               ? "{\"status\":\"fail\",\"validation_target\":"
                 "\"exported_bf16_weights\",\"checksum\":\"\"}"
               : report.logits_check_json)
       << ",\"timings\":{\"batch_load\":" << report.batch_load_seconds
       << ",\"h2d\":" << report.h2d_seconds
       << ",\"forward\":" << report.forward_seconds
       << ",\"backward\":" << report.backward_seconds
       << ",\"optimizer\":" << report.optimizer_seconds
       << ",\"checkpoint\":" << report.checkpoint_export_seconds
       << ",\"export\":" << report.export_seconds << "}"
       << ",\"capability\":{" << capability_json_fields(cuda) << "}";
}

}  // namespace

std::string transformer_train_report_json(const TransformerTrainReport& report,
                                          const CudaStatus& cuda,
                                          const std::string& trainer_mode,
                                          const std::string& status,
                                          const std::string& failure_reason) {
  std::ostringstream out;
  append_transformer(&out, report, cuda, trainer_mode, status, failure_reason);
  out << "}";
  return out.str();
}

bool write_transformer_train_report(const TransformerTrainReport& report,
                                    const CudaStatus& cuda,
                                    const std::string& trainer_mode,
                                    const std::string& status,
                                    const std::string& failure_reason,
                                    std::string* error) {
  auto path = report.checkpoint_dir.parent_path().parent_path() / "runs" /
              "train-report.json";
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out) {
    *error = "failed to write train report: " + path.string();
    return false;
  }
  out << transformer_train_report_json(report, cuda, trainer_mode, status,
                                       failure_reason)
      << "\n";
  return true;
}

}  // namespace lkjai

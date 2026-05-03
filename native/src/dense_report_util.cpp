#include "dense_report_util.hpp"

#include <fstream>

#include "artifact.hpp"
#include "json_min.hpp"

namespace lkjai {

std::string train_report_file_digest(const std::filesystem::path& path) {
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

std::string train_report_packed_cache_digest(const std::filesystem::path& dir) {
  uint64_t hash = 1469598103934665603ull;
  for (const auto& name :
       {"metadata.json", "tokens.bin", "loss_mask.bin", "starts.bin"}) {
    auto path = dir / name;
    hash = (hash ^ artifact_text_checksum(name)[0]) * 1099511628211ull;
    if (!std::filesystem::is_regular_file(path)) continue;
    auto text = train_report_file_digest(path);
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

std::string train_report_manifest_checksum(const std::filesystem::path& dir) {
  auto manifest = read_text(dir / "manifest.json");
  auto checksum = json_first_string(manifest, "weights_checksum");
  return checksum.empty() ? train_report_file_digest(dir / "weights.lkjw")
                          : checksum;
}

long long dense_report_parameter_count(const DenseTrainReport& report) {
  auto config = read_text(report.config_path);
  long long vocab = json_int_value(config, "vocab_size", 0);
  long long hidden = json_int_value(config, "hidden_size", 0);
  return 2 * vocab * hidden;
}

void append_dense_loss_samples(std::ostringstream* out,
                               const std::vector<DenseLossSample>& samples) {
  *out << "[";
  for (size_t i = 0; i < samples.size(); ++i) {
    if (i > 0) *out << ",";
    *out << "{\"step\":" << samples[i].step << ",\"loss\":"
         << samples[i].loss << "}";
  }
  *out << "]";
}

void append_dense_tuning_fields(std::ostringstream* out,
                                const DenseTrainReport& report) {
  *out << ",\"dense_autotune_enabled\":"
       << (report.dense_autotune_enabled ? "true" : "false")
       << ",\"dense_autotune_mode\":\""
       << json_escape(report.dense_autotune_mode) << "\""
       << ",\"dense_workspace_sweep_bytes\":\""
       << json_escape(report.dense_workspace_sweep_bytes) << "\""
       << ",\"dense_cublaslt_logits_algo_id\":"
       << report.dense_cublaslt_logits_algo_id
       << ",\"dense_cublaslt_head_grad_algo_id\":"
       << report.dense_cublaslt_head_grad_algo_id
       << ",\"dense_cublaslt_hidden_grad_algo_id\":"
       << report.dense_cublaslt_hidden_grad_algo_id
       << ",\"dense_cublaslt_logits_workspace_bytes\":"
       << report.dense_cublaslt_logits_workspace_bytes
       << ",\"dense_cublaslt_head_grad_workspace_bytes\":"
       << report.dense_cublaslt_head_grad_workspace_bytes
       << ",\"dense_cublaslt_hidden_grad_workspace_bytes\":"
       << report.dense_cublaslt_hidden_grad_workspace_bytes
       << ",\"dense_allocator_backend\":\""
       << json_escape(report.dense_allocator_backend) << "\""
       << ",\"dense_async_alloc_supported\":"
       << (report.dense_async_alloc_supported ? "true" : "false")
       << ",\"dense_mempool_release_threshold_bytes\":"
       << report.dense_mempool_release_threshold_bytes
       << ",\"dense_workspace_high_water_bytes\":"
       << report.dense_workspace_high_water_bytes
       << ",\"dense_workspace_reallocations\":"
       << report.dense_workspace_reallocations
       << ",\"dense_timing_mode\":\"" << json_escape(report.dense_timing_mode)
       << "\",\"dense_head_f32_cache_enabled\":"
       << (report.dense_head_f32_cache_enabled ? "true" : "false")
       << ",\"dense_head_f32_cache_refreshes\":"
       << report.dense_head_f32_cache_refreshes;
}

}  // namespace lkjai

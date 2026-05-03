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

}  // namespace lkjai

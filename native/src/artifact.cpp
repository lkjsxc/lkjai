#include "artifact.hpp"

#include <iomanip>
#include <fstream>
#include <sstream>
#include <string_view>

#include "artifact_manifest.hpp"
#include "artifact_validate.hpp"
#include "json_min.hpp"

namespace lkjai {
namespace {

const char* kRequired[] = {
    "manifest.json",
    "config.json",
    "tokenizer.json",
    "weights.index.json",
    "weights.lkjw",
};

bool file_exists(const std::filesystem::path& path) {
  return std::filesystem::is_regular_file(path);
}

}  // namespace

ArtifactStatus load_artifact(const std::filesystem::path& root,
                             const std::string& model_name) {
  ArtifactStatus status;
  status.model_name = model_name;
  status.model_dir = root / model_name;
  for (const char* name : kRequired) {
    if (!file_exists(status.model_dir / name)) {
      status.missing.push_back(name);
    }
  }
  if (!status.missing.empty()) {
    status.error = "missing native artifact files";
    return status;
  }
  if (std::string error; !inspect_artifact(status.model_dir, &error)) {
    status.error = error;
    return status;
  }
  status.loaded = true;
  return status;
}

bool inspect_artifact(const std::filesystem::path& model_dir,
                      std::string* error) {
  const auto manifest = model_dir / "manifest.json";
  const auto index = model_dir / "weights.index.json";
  const auto weights = model_dir / "weights.lkjw";
  const auto config = model_dir / "config.json";
  const auto tokenizer = model_dir / "tokenizer.json";
  auto manifest_text = read_text(manifest);
  auto config_text = read_text(config);
  auto tokenizer_text = read_text(tokenizer);
  std::string artifact_kind;
  if (!validate_manifest(manifest_text, config_text, tokenizer_text, &artifact_kind,
                         error)) {
    return false;
  }
  auto kind = json_first_string(manifest_text, "kind");
  if (kind != "dense" && kind != "transformer") {
    *error = "manifest kind must be dense or transformer";
    return false;
  }
  if (artifact_kind == "checkpoint" &&
      (!file_exists(model_dir / "optimizer.index.json") ||
       !file_exists(model_dir / "optimizer.lkjw"))) {
    *error = "checkpoint artifact missing optimizer files";
    return false;
  }
  if (kind == "dense" && artifact_kind == "checkpoint" &&
      !validate_dense_optimizer(model_dir, error)) {
    return false;
  }
  auto index_text = read_text(index);
  auto weight_bytes = std::filesystem::file_size(weights);
  if (weight_bytes == 0) {
    *error = "weights.lkjw is empty";
    return false;
  }
  if (kind == "dense")
    return validate_dense_weight_index(index_text, weight_bytes, error);
  if (!validate_transformer_weight_index(index_text, config_text, weight_bytes,
                                         error)) return false;
  return true;
}

std::string artifact_logits_checksum(const std::filesystem::path& model_dir) {
  std::ifstream file(model_dir / "weights.lkjw", std::ios::binary);
  uint64_t hash = 1469598103934665603ull;
  char ch = 0;
  uint64_t read = 0;
  while (read < 65536 && file.get(ch)) {
    hash = (hash ^ static_cast<unsigned char>(ch)) * 1099511628211ull;
    ++read;
  }
  std::ostringstream out;
  out << std::hex << hash;
  return out.str();
}
}  // namespace lkjai

#include "artifact.hpp"

#include <fstream>

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
  auto manifest_text = read_text(manifest);
  if (!contains_json_string(manifest_text, "format",
                            "lkjai-native-artifact-v1")) {
    *error = "manifest format must be lkjai-native-artifact-v1";
    return false;
  }
  auto index_text = read_text(index);
  if (index_text.find("\"tensors\"") == std::string::npos) {
    *error = "weights.index.json missing tensors field";
    return false;
  }
  if (std::filesystem::file_size(weights) == 0) {
    *error = "weights.lkjw is empty";
    return false;
  }
  return true;
}

}  // namespace lkjai

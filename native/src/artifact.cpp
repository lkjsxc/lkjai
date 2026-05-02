#include "artifact.hpp"

#include <cctype>
#include <fstream>
#include <string_view>

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

bool read_u64_after(std::string_view text, std::string_view key, size_t start,
                    uint64_t* out) {
  auto pos = text.find(key, start);
  if (pos == std::string_view::npos) return false;
  pos = text.find(':', pos + key.size());
  if (pos == std::string_view::npos) return false;
  ++pos;
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  try {
    *out = static_cast<uint64_t>(std::stoull(std::string(text.substr(pos))));
    return true;
  } catch (...) {
    return false;
  }
}

bool validate_weight_index(std::string_view text, uint64_t weight_bytes,
                           std::string* error) {
  if (text.find("\"tensors\"") == std::string_view::npos) {
    *error = "weights.index.json missing tensors field";
    return false;
  }
  size_t pos = 0;
  int tensors = 0;
  while ((pos = text.find("\"byte_offset\"", pos)) != std::string_view::npos) {
    uint64_t offset = 0;
    uint64_t length = 0;
    if (!read_u64_after(text, "\"byte_offset\"", pos, &offset) ||
        !read_u64_after(text, "\"byte_length\"", pos, &length)) {
      *error = "weights.index.json has invalid tensor offsets";
      return false;
    }
    if (offset % 256 != 0) {
      *error = "weights.index.json tensor offset is not 256-byte aligned";
      return false;
    }
    if (length == 0 || offset > weight_bytes || length > weight_bytes - offset) {
      *error = "weights.index.json tensor range exceeds weights.lkjw";
      return false;
    }
    auto shape = text.rfind("\"shape\"", pos);
    if (shape == std::string_view::npos) {
      *error = "weights.index.json tensor missing shape";
      return false;
    }
    ++tensors;
    pos += 13;
  }
  if (tensors == 0) {
    *error = "weights.index.json contains no tensor entries";
    return false;
  }
  return true;
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
  auto weight_bytes = std::filesystem::file_size(weights);
  if (weight_bytes == 0) {
    *error = "weights.lkjw is empty";
    return false;
  }
  if (!validate_weight_index(index_text, weight_bytes, error)) return false;
  return true;
}

}  // namespace lkjai

#include "artifact.hpp"

#include <cctype>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string_view>

#include "artifact_manifest.hpp"
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

bool require_tensor(std::string_view text, const std::string& name,
                    std::string* error) {
  if (contains_json_string(text, "name", name)) return true;
  *error = "weights.index.json missing tensor " + name;
  return false;
}

bool valid_dtype(std::string_view text, size_t start) {
  for (auto dtype : {"u16", "u32", "f16", "bf16", "f32"}) {
    if (contains_json_string(text.substr(start), "dtype", dtype)) return true;
  }
  return false;
}

bool require_entry_metadata(std::string_view text, size_t pos,
                            std::string* error) {
  auto start = text.rfind('{', pos);
  if (start == std::string_view::npos) start = 0;
  auto end = text.find('}', pos);
  auto entry = text.substr(start, end == std::string_view::npos ? text.size()
                                                                : end - start);
  if (entry.find("\"name\"") == std::string_view::npos) {
    *error = "weights.index.json tensor missing name";
    return false;
  }
  if (!valid_dtype(entry, 0)) {
    *error = "weights.index.json tensor missing supported dtype";
    return false;
  }
  if (entry.find("\"shape\"") == std::string_view::npos) {
    *error = "weights.index.json tensor missing shape";
    return false;
  }
  return true;
}

bool validate_weight_index(std::string_view text, std::string_view config,
                           uint64_t weight_bytes, std::string* error) {
  if (text.find("\"tensors\"") == std::string_view::npos) {
    *error = "weights.index.json missing tensors field";
    return false;
  }
  if (!require_tensor(text, "tok_embeddings", error)) return false;
  int layers = json_int_value(config, "layers", 1);
  for (int layer = 0; layer < layers; ++layer) {
    auto p = "layers." + std::to_string(layer) + ".";
    for (const auto& name : {p + "attn.q_proj", p + "attn.k_proj",
                             p + "attn.v_proj", p + "attn.o_proj",
                             p + "mlp.gate_proj", p + "mlp.up_proj",
                             p + "mlp.down_proj", p + "attn_norm",
                             p + "mlp_norm"}) {
      if (!require_tensor(text, name, error)) return false;
    }
  }
  if (!require_tensor(text, "final_norm", error)) return false;
  if (!require_tensor(text, "lm_head", error)) return false;
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
    if (!require_entry_metadata(text, pos, error)) return false;
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
  const auto config = model_dir / "config.json";
  const auto tokenizer = model_dir / "tokenizer.json";
  auto manifest_text = read_text(manifest);
  auto config_text = read_text(config);
  auto tokenizer_text = read_text(tokenizer);
  std::string kind;
  if (!validate_manifest(manifest_text, config_text, tokenizer_text, &kind,
                         error)) {
    return false;
  }
  if (kind == "checkpoint" &&
      (!file_exists(model_dir / "optimizer.index.json") ||
       !file_exists(model_dir / "optimizer.lkjw"))) {
    *error = "checkpoint artifact missing optimizer files";
    return false;
  }
  auto index_text = read_text(index);
  auto weight_bytes = std::filesystem::file_size(weights);
  if (weight_bytes == 0) {
    *error = "weights.lkjw is empty";
    return false;
  }
  if (!validate_weight_index(index_text, config_text, weight_bytes, error)) {
    return false;
  }
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

#include "artifact_validate.hpp"

#include <cctype>

#include "json_min.hpp"

namespace lkjai {
namespace {

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

bool valid_dtype(std::string_view text) {
  for (auto dtype : {"u16", "u32", "f16", "bf16", "f32"}) {
    if (contains_json_string(text, "dtype", dtype)) return true;
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
  if (!valid_dtype(entry)) {
    *error = "weights.index.json tensor missing supported dtype";
    return false;
  }
  if (entry.find("\"shape\"") == std::string_view::npos) {
    *error = "weights.index.json tensor missing shape";
    return false;
  }
  return true;
}

bool validate_tensor_ranges(std::string_view text, uint64_t weight_bytes,
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
    if (offset % 256 != 0 || length == 0 || offset > weight_bytes ||
        length > weight_bytes - offset) {
      *error = "weights.index.json tensor range is invalid";
      return false;
    }
    if (!require_entry_metadata(text, pos, error)) return false;
    ++tensors;
    pos += 13;
  }
  if (tensors == 0) *error = "weights.index.json contains no tensor entries";
  return tensors > 0;
}

}  // namespace

bool validate_dense_weight_index(std::string_view text, uint64_t weight_bytes,
                                 std::string* error) {
  return validate_tensor_ranges(text, weight_bytes, error) &&
         require_tensor(text, "tok_embeddings", error) &&
         require_tensor(text, "lm_head", error);
}

bool validate_transformer_weight_index(std::string_view text,
                                       std::string_view config,
                                       uint64_t weight_bytes,
                                       std::string* error) {
  if (!validate_tensor_ranges(text, weight_bytes, error) ||
      !require_tensor(text, "tok_embeddings", error)) {
    return false;
  }
  bool decoder = contains_json_string(config, "model_kind", "decoder");
  if (!decoder && !require_tensor(text, "pos_embeddings", error)) return false;
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
  bool tied = config.find("\"tie_embeddings\":true") != std::string_view::npos;
  return require_tensor(text, "final_norm", error) &&
         (tied || require_tensor(text, "lm_head", error));
}

bool validate_dense_optimizer(const std::filesystem::path& model_dir,
                              std::string* error) {
  auto text = read_text(model_dir / "optimizer.index.json");
  for (const auto& name :
       {"master.tok_embeddings", "adam_m.tok_embeddings",
        "adam_v.tok_embeddings", "master.lm_head", "adam_m.lm_head",
        "adam_v.lm_head"}) {
    if (!require_tensor(text, name, error)) return false;
  }
  return true;
}

bool validate_transformer_optimizer(const std::filesystem::path& model_dir,
                                    std::string_view config,
                                    std::string* error) {
  auto text = read_text(model_dir / "optimizer.index.json");
  auto require_triplet = [&](const std::string& name) {
    return require_tensor(text, "master." + name, error) &&
           require_tensor(text, "adam_m." + name, error) &&
           require_tensor(text, "adam_v." + name, error);
  };
  bool decoder = contains_json_string(config, "model_kind", "decoder");
  bool tied = config.find("\"tie_embeddings\":true") != std::string_view::npos;
  if (!require_triplet("tok_embeddings")) {
    return false;
  }
  if (!decoder && !require_triplet("pos_embeddings")) return false;
  int layers = json_int_value(config, "layers", 1);
  for (int layer = 0; layer < layers; ++layer) {
    auto p = "layers." + std::to_string(layer) + ".";
    for (const auto& name : {p + "attn.q_proj", p + "attn.k_proj",
                             p + "attn.v_proj", p + "attn.o_proj",
                             p + "mlp.gate_proj", p + "mlp.up_proj",
                             p + "mlp.down_proj", p + "attn_norm",
                             p + "mlp_norm"}) {
      if (!require_triplet(name)) return false;
    }
  }
  return require_triplet("final_norm") && (tied || require_triplet("lm_head"));
}

}  // namespace lkjai
